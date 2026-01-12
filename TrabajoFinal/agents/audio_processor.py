"""
Módulo de procesamiento de audio
Incluye funciones para preprocesar audios antes de la transcripción
"""
import numpy as np
from pydub import AudioSegment , effects
from pydub.silence import detect_nonsilent
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List
import logging
import torch
import sys
import os

# Añadir el directorio raíz al path para importar config si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import config
except ImportError:
    # Si falla, intentar una importación relativa directa si estamos en el paquete agents
    try:
        from .. import config
    except ImportError:
        logging.warning("No se pudo importar config, se usarán valores por defecto")
        class ConfigMock:
            SAMPLE_RATE = 16000
        config = ConfigMock()

from df.enhance import enhance, init_df, load_audio, save_audio
from scipy.io import wavfile
from scipy import signal
from pydub.effects import normalize, compress_dynamic_range
import io
import time
import tempfile
from scipy.ndimage import binary_closing, binary_dilation


class AudioProcessor:
    """Procesador de audio para limpieza y preparación"""
    
    def __init__(self, sample_rate: int = getattr(config, 'SAMPLE_RATE', 16000)):
        self.sample_rate = sample_rate
        # Inicializar el modelo de DeepFilterNet una sola vez
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model, self.df_state, _ = init_df()
            self.model.to(self.device)
            logging.info(f"DeepFilterNet inicializado correctamente en: {self.device}")
        except Exception as e:
            logging.error(f"Error inicializando DeepFilterNet: {e}")
            self.model, self.df_state = None, None
        
    def load_audio_OLD(self, audio_path: str) -> AudioSegment:
        """
        Carga un archivo de audio en formato AudioSegment
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            AudioSegment con el audio cargado
        """
        # Convertir a Path para manejar correctamente rutas con espacios en Windows
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de audio: {audio_path}")
        
        # Para archivos WAV, usar soundfile que no requiere ffmpeg y soporta MULAW
        # Para archivos WAV, intentar primero con soundfile (más rápido, soporta MULAW)
        if audio_path.suffix.lower() == '.wav':
            try:
                # Leer archivo WAV con soundfile
                audio_data, sample_rate = sf.read(str(audio_path), dtype='int16')
                
                # Si es estéreo, convertir a mono
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1).astype(np.int16)
                
                # Crear AudioSegment desde raw data
                audio = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1
                )
                
                print(f"Audio cargado (soundfile): {len(audio)/1000:.1f}s, {sample_rate}Hz")
                
            except Exception as e:
                # No es un error crítico, solo un formato que soundfile no maneja
                # print(f"Info: soundfile no pudo leer el archivo ({e}), intentando con scipy...")
                try:
                    # Fallback 1: scipy.io.wavfile (no requiere ffmpeg)
                    sample_rate, audio_data = wavfile.read(str(audio_path))
                    
                    # Si es estéreo, convertir a mono
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1).astype(np.int16)
                    elif audio_data.dtype != np.int16:
                        # Convertir a int16 si es necesario
                        if audio_data.dtype == np.float32:
                            audio_data = (audio_data * 32767).astype(np.int16)
                        else:
                            audio_data = audio_data.astype(np.int16)

                    audio = AudioSegment(
                        audio_data.tobytes(),
                        frame_rate=sample_rate,
                        sample_width=2,
                        channels=1
                    )
                    print(f"Audio cargado (scipy): {len(audio)/1000:.1f}s, {sample_rate}Hz")
                    
                except Exception as e2:
                    # print(f"Info: scipy no pudo leer el archivo ({e2}), intentando con pydub...")
                    try:
                        # Fallback 2: intentar con pydub (usa ffmpeg/avconv, más robusto)
                        audio = AudioSegment.from_file(str(audio_path))
                        print(f"Audio cargado (pydub): {len(audio)/1000:.1f}s, {audio.frame_rate}Hz")
                    except Exception as e3:
                        print(f"Error fatal al cargar WAV: {e3}")
                        raise e
        else:
            # Para otros formatos, intentar con pydub (requiere ffmpeg)
            audio = AudioSegment.from_file(str(audio_path))
        
        # Convertir a mono si es estéreo (por si acaso)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        return audio


    def load_audio(self, audio_path: str) -> AudioSegment:
        """
        Carga cualquier archivo de audio soportado en un AudioSegment mono.
        """
        path = Path(audio_path)
        
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de audio: {path}")

        try:
            # 'from_file' detecta automáticamente si es WAV, MP3, OGG, etc.
            # Es la forma única y estándar de pydub para cargar archivos.
            audio = AudioSegment.from_file(str(path))
            
            # Unificamos el post-procesado: siempre devolver en Mono
            #if audio.channels > 1:
            #   audio = audio.set_channels(1)
                
            print(f"Audio cargado exitosamente: {path.name} ({len(audio)/1000:.1f}s, {audio.frame_rate}Hz)")
            return audio

        except Exception as e:
            raise RuntimeError(f"Error al cargar el audio {path.name}: {e}")

    def detect_hold_tones(self, audio: AudioSegment) -> List[Tuple[float, float]]:
        SR = audio.frame_rate
        HOP_LENGTH = 512
        DURATION_THRESHOLD = 5.0 # La música debe durar al menos 5s
        
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if len(samples) == 0: return []
        samples /= (np.max(np.abs(samples)) + 1e-6)

        zcr = librosa.feature.zero_crossing_rate(y=samples, hop_length=HOP_LENGTH)[0]
        flatness = librosa.feature.spectral_flatness(y=samples, hop_length=HOP_LENGTH)[0]
        rms = librosa.feature.rms(y=samples, hop_length=HOP_LENGTH)[0]
        rms_norm = rms / (np.max(rms) + 1e-6)

        # Criterio estricto de música (tonos puros y estables)
        is_hold = (zcr < 0.07) & (flatness < 0.03) & (rms_norm > 0.15)

        # --- NUEVA PROTECCIÓN DE BIENVENIDA ---
        # Forzamos que los primeros 4 segundos NUNCA se marquen como música
        safe_zone_frames = int(4.0 * SR / HOP_LENGTH)
        is_hold[:safe_zone_frames] = False

        gap_frames = int(1.5 * SR / HOP_LENGTH)
        refined_mask = binary_closing(is_hold, structure=np.ones(gap_frames))
        refined_mask = binary_dilation(refined_mask, structure=np.ones(int(0.5 * SR / HOP_LENGTH)), iterations=-1)

        times = librosa.frames_to_time(np.arange(len(refined_mask)), sr=SR, hop_length=HOP_LENGTH)
        segments = []
        in_segment = False
        start_time = 0
        
        for i, active in enumerate(refined_mask):
            if active and not in_segment:
                start_time = times[i]
                in_segment = True
            elif not active and in_segment:
                end_time = times[i]
                if end_time - start_time >= DURATION_THRESHOLD:
                    # Dejamos 1 segundo extra de margen al final del segmento
                    # para asegurar que no se corte el primer "Hola"
                    segments.append((start_time, max(start_time, end_time - 1.0)))
                in_segment = False
                
        return segments

    def remove_hold_tones(self, audio: AudioSegment) -> Tuple[AudioSegment, List[Tuple[float, float]]]:
        # 1. Obtenemos los segmentos de TONOS (lo que queremos quitar)
        hold_tone_segments = self.detect_hold_tones(audio)
        
        if not hold_tone_segments:
            return audio, []

        # 2. Reconstrucción: Solo conservamos lo que NO está en hold_tone_segments
        cleaned_audio = AudioSegment.empty()
        current_pos_ms = 0
        total_duration_ms = len(audio)
        
        # Ordenamos por si acaso para evitar saltos temporales
        hold_tone_segments.sort()

        for start_sec, end_sec in hold_tone_segments:
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            # Si hay espacio entre el final del último tono y el inicio de este, es VOZ
            if start_ms > current_pos_ms:
                # Extraemos la voz
                voice_segment = audio[current_pos_ms:start_ms]
                
                # Unimos con un pequeño crossfade si ya hay audio previo
                if len(cleaned_audio) > 0:
                    cleaned_audio = cleaned_audio.append(voice_segment, crossfade=50)
                else:
                    cleaned_audio = voice_segment
            
            # Saltamos el tono
            current_pos_ms = end_ms

        # 3. No olvides el trozo de audio después del último tono detectado
        if current_pos_ms < total_duration_ms:
            remaining_voice = audio[current_pos_ms:]
            if len(cleaned_audio) > 0:
                cleaned_audio = cleaned_audio.append(remaining_voice, crossfade=50)
            else:
                cleaned_audio = remaining_voice

        return cleaned_audio, hold_tone_segments

    def apply_bandpass_filter_OLD(self, audio: AudioSegment, 
                            low_cut: int = 300, 
                            high_cut: int = 3400, 
                            order: int = 4) -> AudioSegment:
        """
        Aplica un filtro paso banda suave para mejorar la claridad de la voz 
        evitando distorsiones por gating agresivo.
        """
        try:
            # 1. Configuración del filtro Butterworth
            fs = audio.frame_rate
            nyquist = 0.5 * fs
            low = max(0.001, low_cut / nyquist)
            high = min(0.999, high_cut / nyquist)
            sos = signal.butter(order, [low, high], btype='band', output='sos')

            # 2. Conversión a numpy (trabajamos en float para evitar clipping intermedio)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            
            # Normalización suave de entrada si está muy bajo (evita ruido de cuantización)
            if audio.max_dBFS < -20:
                max_sample = np.max(np.abs(samples))
                if max_sample > 0:
                    samples = samples / max_sample * 25000 # Escala segura

            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                filtered = np.zeros_like(samples)
                for i in range(audio.channels):
                    filtered[:, i] = signal.sosfiltfilt(sos, samples[:, i])
            else:
                filtered = signal.sosfiltfilt(sos, samples)

            # 3. NOISE FLOOR (En lugar de Hard Gate, usamos una reducción suave)
            # Solo silenciamos lo que es verdaderamente insignificante (< 1%)
            # Un gate del 5% es demasiado agresivo y corta palabras.
            noise_threshold = np.max(np.abs(filtered)) * 0.01 
            filtered[np.abs(filtered) < noise_threshold] = 0 

            # 4. Limitación suave y reconversión
            max_val = float(2**(8 * audio.sample_width - 1))
            
            # Si el filtro causó picos (overshoot), los normalizamos proporcionalmente
            current_max = np.max(np.abs(filtered))
            if current_max >= max_val:
                filtered = filtered * (max_val - 1) / current_max
            
            filtered = filtered.astype(np.int16)
            processed_audio = audio._spawn(filtered.tobytes())

            # 5. COMPRESIÓN DINÁMICA moderada para nivelar voces
            # Usamos valores más estándar para evitar el efecto "bombeo"
            compressed = effects.compress_dynamic_range(
                processed_audio, 
                threshold=-18.0, 
                ratio=3.0,       
                attack=5.0, 
                release=50.0   
            )
            
            # Normalización final al 90% para evitar clipping en reproductores
            return effects.normalize(compressed, headroom=1.0)

        except Exception as e:
            logging.error(f"Error en apply_bandpass_filter: {e}")
            return audio # Fallback al audio original si algo falla


    def apply_bandpass_filterFUN(self, audio: AudioSegment, 
                                low_cut: int = 300, 
                                high_cut: int = 3400, 
                                order: int = 2) -> AudioSegment: # Orden 2 o 3 es más natural
        """
        Aplica un filtro paso banda y mejora la voz sin efectos robóticos.
        """
        try:
            # 1. Configuración del filtro (Bajamos el orden para que sea menos 'quirúrgico')
            fs = audio.frame_rate
            nyquist = 0.5 * fs
            low = max(0.001, low_cut / nyquist)
            high = min(0.999, high_cut / nyquist)
            
            # Usamos SOS para estabilidad numérica
            sos = signal.butter(order, [low, high], btype='band', output='sos')

            # 2. Conversión a float32 normalizada (-1.0 a 1.0)
            # Esto es clave para evitar distorsión matemática
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            max_possible_val = float(2**(8 * audio.sample_width - 1))
            samples = samples / max_possible_val

            # Aplicar filtro por canales
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                filtered = np.zeros_like(samples)
                for i in range(audio.channels):
                    filtered[:, i] = signal.sosfiltfilt(sos, samples[:, i])
            else:
                filtered = signal.sosfiltfilt(sos, samples)

            # 3. ELIMINADO EL GATE AGRESIVO. 
            # Si quieres reducir ruido, es mejor usar efectos de expansión suave 
            # o simplemente dejar que el filtro haga su trabajo.

            # 4. Normalización y reconversión segura
            # Evitamos el clipping limitando el rango antes de convertir
            filtered = np.clip(filtered, -1.0, 1.0)
            final_samples = (filtered * (max_possible_val - 1)).astype(np.int16)

            # Reconstruir el AudioSegment
            processed_audio = audio._spawn(final_samples.tobytes())

            # 5. COMPRESIÓN DINÁMICA (Ajustada para naturalidad)
            # Un release más largo (200ms) evita el efecto de "bombeo" o vibración
            compressed = effects.compress_dynamic_range(
                processed_audio, 
                threshold=-20.0, 
                ratio=2.5,      
                attack=10.0,   # Attack ligeramente más lento para preservar transientes
                release=200.0  # Release más largo = sonido más natural
            )
            
            return effects.normalize(compressed, headroom=0.5)

        except Exception as e:
            logging.error(f"Error en apply_bandpass_filter: {e}")
        return audio

    def apply_bandpass_filter_ADAPTADO(self, audio: AudioSegment, 
                                low_cut: int = 300, 
                                high_cut: int = 3400, 
                                order: int = 2) -> AudioSegment:
        """
        Versión optimizada: Elimina chillidos, siseos y evita el efecto robótico
        manteniendo la compatibilidad con el nombre de función original.
        """
        try:
            # 1. Configuración de parámetros y conversión a Float
            fs = audio.frame_rate
            nyq = 0.5 * fs
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            max_possible_val = float(2**(8 * audio.sample_width - 1))
            samples = samples / max_possible_val # Normalización a rango -1.0 a 1.0

            # --- 2. DETECCIÓN Y ELIMINACIÓN DE CHILLIDOS (Notch Filter) ---
            # Analizamos frecuencias para encontrar picos metálicos (resonancias)
            freqs, psd = signal.welch(samples, fs, nperseg=2048)
            mask = (freqs > 1000) & (freqs < 4000) # Rango donde suelen estar los silbidos
            
            if np.any(mask):
                mean_psd = np.mean(psd[mask])
                # Si un pico es 15 veces más fuerte que el promedio, lo eliminamos
                peaks, _ = signal.find_peaks(psd[mask], height=mean_psd * 15)
                
                for peak in peaks:
                    target_freq = freqs[mask][peak]
                    # Filtro muesca (Notch) muy estrecho
                    b, a = signal.iirnotch(target_freq / nyq, Q=30)
                    if audio.channels > 1:
                        samples = samples.reshape((-1, audio.channels))
                        for i in range(audio.channels):
                            samples[:, i] = signal.filtfilt(b, a, samples[:, i])
                        samples = samples.flatten()
                    else:
                        samples = signal.filtfilt(b, a, samples)

            # --- 3. ELIMINACIÓN DE SISEO (De-esser / Suavizado de 'S') ---
            # Aplicamos una caída suave a las frecuencias muy altas
            b_deess, a_deess = signal.cheby1(1, 4, 2800 / nyq, btype='low')
            samples = signal.filtfilt(b_deess, a_deess, samples)

            # --- 4. FILTRO PASO BANDA (Filtro solicitado) ---
            # Usamos SOS y filtfilt para evitar el desfase que causa el sonido robótico
            sos = signal.butter(order, [low_cut / nyq, high_cut / nyq], btype='band', output='sos')
            
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                filtered = np.zeros_like(samples)
                for i in range(audio.channels):
                    filtered[:, i] = signal.sosfiltfilt(sos, samples[:, i])
                samples = filtered.flatten()
            else:
                samples = signal.sosfiltfilt(sos, samples)

            # --- 5. RECONSTRUCCIÓN Y NIVELACIÓN ---
            samples = np.clip(samples, -1.0, 1.0)
            final_samples = (samples * (max_possible_val - 1)).astype(np.int16)
            processed_audio = audio._spawn(final_samples.tobytes())

            # Compresión dinámica moderada (para que la voz suene constante y profesional)
            compressed = effects.compress_dynamic_range(
                processed_audio, 
                threshold=-22.0, 
                ratio=2.0,      
                attack=10.0, 
                release=200.0   
            )
            
            # Normalización final con margen de seguridad
            return effects.normalize(compressed, headroom=0.5)

        except Exception as e:
            logging.error(f"Error en apply_bandpass_filter: {e}")
            return audio

    
    def apply_bandpass_filter(self, audio: AudioSegment) -> AudioSegment:
        """
        Versión con IA corregida: Evita voz de ardilla y pérdida de voz
        mediante normalización previa y control estricto de sample rate.
        """
        try:
            # 1. ESTANDARIZACIÓN TOTAL (Evita errores de interpretación)
            # Forzamos a 16-bit, Mono, 48kHz antes de cualquier cálculo
            original_sr = audio.frame_rate
            audio_working = audio.set_sample_width(2).set_channels(1).set_frame_rate(48000)
            
            # Normalizamos volumen de entrada para que la IA "escuche" bien
            audio_working = effects.normalize(audio_working, headroom=0.1)

            # 2. CONVERSIÓN A FLOAT32 (Rango -1.0 a 1.0)
            # Usamos np.frombuffer que es más seguro para evitar ruidos de "aliasing"
            samples = np.frombuffer(audio_working.raw_data, dtype=np.int16).astype(np.float32)
            samples /= 32768.0  # Normalización exacta para 16-bit int

            # 3. PROCESAMIENTO IA
            # Importante: Pasamos el tensor en CPU ya que la función enhance de DeepFilterNet 
            # suele manejar internamente el movimiento al dispositivo del modelo y puede 
            # intentar conversiones a numpy que fallan si el tensor ya está en CUDA.
            audio_tensor = torch.from_numpy(samples).unsqueeze(0) # Mantener en CPU para enhance()
            
            with torch.no_grad():
                enhanced_audio = enhance(self.model, self.df_state, audio_tensor)

            # 4. RECONSTRUCCIÓN CRÍTICA
            # Aseguramos que el resultado sea un tensor de CPU antes de convertir a numpy
            if torch.is_tensor(enhanced_audio):
                enhanced_samples = enhanced_audio.detach().cpu().numpy().flatten()
            else:
                # Fallback si enhance devuelve algo que no es un tensor (lista/tupla)
                enhanced_samples = np.array(enhanced_audio).flatten()

            # --- 5. REDUCCIÓN DE GRAVES (ECUALIZACIÓN) ---
            # Aplicamos un filtro paso alto a 180Hz para quitar el retumbo
            # Si quieres menos graves aún, sube 180 a 250.
            nyq = 0.5 * 48000
            cutoff = 250  # Hz
            b, a = signal.butter(2, cutoff / nyq, btype='highpass')
            enhanced_samples = signal.filtfilt(b, a, enhanced_samples)
            
            # Limitar para evitar "clipping" (estática demoníaca)
            enhanced_samples = np.clip(enhanced_samples, -1.0, 1.0)
            
            # Convertir de nuevo a Int16 con precisión
            final_samples = (enhanced_samples * 32767.0).astype(np.int16)

            # 5. CREACIÓN DEL OBJETO DE AUDIO (Sin usar _spawn interno)
            # Creamos un nuevo objeto desde los bytes para asegurar frescura de metadatos
            processed_audio = AudioSegment(
                data=final_samples.tobytes(),
                sample_width=2,
                frame_rate=48000,
                channels=1
            )

            # 6. RETORNO AL FORMATO ORIGINAL (Si es necesario)
            if original_sr != 48000:
                processed_audio = processed_audio.set_frame_rate(original_sr)

            # Normalización final para consistencia
            return effects.normalize(processed_audio, headroom=0.5)

        except Exception as e:
            logging.error(f"Error crítico en IA: {e}")
            # FALLBACK: Si la IA falla, aplicamos un filtro básico para no perder el audio
            return self._basic_fallback_filter(audio)

    def _basic_fallback_filter(self, audio: AudioSegment) -> AudioSegment:
        """Filtro de emergencia si la IA falla o el audio es irreconocible"""
        return effects.normalize(audio.low_pass_filter(3500).high_pass_filter(300))

    def remove_silence(self, audio: AudioSegment, 
                min_silence_len: int = 1000,
                silence_thresh_offset: int = -30, # Un poco más bajo para no cortar palabras
                keep_silence: int = 300) -> Tuple[AudioSegment, float]:
        """
        Versión optimizada: No vuelve a normalizar el audio para evitar subir el siseo.
        """
        # Usamos el nivel de decibelios actual del audio ya procesado por el filtro
        threshold = audio.dBFS + silence_thresh_offset

        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=threshold
        )
        
        if not nonsilent_ranges:
            return audio, 0.0
        
        cleaned_audio = AudioSegment.empty()
        duration_ms = len(audio)
        
        for start, end in nonsilent_ranges:
            # Aumentamos el padding para evitar cortes abruptos (evita el efecto 'sh')
            start_pad = max(0, start - keep_silence)
            end_pad = min(duration_ms, end + keep_silence)
            
            chunk = audio[start_pad:end_pad]
            
            if len(cleaned_audio) > 0:
                cleaned_audio = cleaned_audio.append(chunk, crossfade=100) # Crossfade más largo para suavidad
            else:
                cleaned_audio = chunk
                
        original_dur = len(audio)
        cleaned_dur = len(cleaned_audio)
        silence_pct = ((original_dur - cleaned_dur) / original_dur) * 100 if original_dur > 0 else 0
        
        return cleaned_audio, silence_pct
    
    def preprocess_audio(self, audio_path: str, output_path: str = None) -> dict:
        """
        Pipeline profesional de preprocesamiento de audio.
        Optimiza la calidad para escucha humana o motores de transcripción (ASR).
        """
        start_time_proc = time.time()
        logging.info(f"Iniciando procesamiento: {audio_path}")
        
        try:
            # 1. Cargar audio
            audio = self.load_audio(audio_path)
            original_duration = len(audio) / 1000.0
            
            # --- ETAPA DE LIMPIEZA DE FRECUENCIAS ---
            # Filtramos primero para que la detección de tonos y silencios 
            # no se confunda con ruidos de baja o alta frecuencia (motores, siseos).
            audio = self.apply_bandpass_filter(audio)
            
            # --- ETAPA DE ELIMINACIÓN DE SEGMENTOS ---
            # A. Eliminar tonos de espera (Usando la versión híbrida música/tonos)
            audio1, hold_tone_segments = self.remove_hold_tones(audio)
            hold_tone_duration = sum(end - start for start, end in hold_tone_segments)
            
            # B. Eliminar silencios (Usando la versión con normalización interna)
            # Se hace después de quitar los tonos para no procesar silencios de la música de espera.
            audio2, silence_percentage = self.remove_silence(audio)
            
            
            final_duration = len(audio) / 1000.0
            
            # Exportar para Whisper
            self.export_for_whisper(audio, str(temp_path))
            # Guardar resultado
            #if output_path:
            #    # Aseguramos formato óptimo para análisis (16kHz, mono es estándar para IA)
            #    audio.export(output_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            
            # Métricas mejoradas
            processing_time = time.time() - start_time_proc
            metrics = {
                "status": "success",
                "original_duration_sec": round(original_duration, 2),
                "final_duration_sec": round(final_duration, 2),
                "hold_tones_detected": len(hold_tone_segments),
                "hold_tone_duration_sec": round(hold_tone_duration, 2),
                "silence_removed_pct": round(silence_percentage, 2),
                "total_time_saved_sec": round(original_duration - final_duration, 2),
                "efficiency_ratio_pct": round((1 - final_duration / original_duration) * 100, 2),
                "processing_execution_time_sec": round(processing_time, 2)
            }
            
            logging.info(f"Procesamiento finalizado en {processing_time:.2f}s")
            return metrics

        except Exception as e:
            logging.error(f"Error procesando {audio_path}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def export_for_whisper(self, audio: AudioSegment, output_path: str):
        """
        Exporta audio optimizado usando pydub (consistente con load_audio).
        """
        try:
            # 1. Estandarizar formato (16kHz, Mono, 16-bit)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            # 2. PROTECCIÓN: Añadir 300ms de silencio al inicio y fin
            # Esto ayuda a Whisper a no "comerse" la primera palabra y estabilizar el VAD.
            padding = AudioSegment.silent(duration=300, frame_rate=16000)
            audio = padding + audio + padding

            # 3. Normalización (Opcional, se mantiene comentada si el usuario lo prefiere)
            # audio = effects.normalize(audio, headroom=0.1)

            # 4. Exportar usando pydub
            # format="wav" asegura compatibilidad máxima con lo que espera Whisper
            audio.export(output_path, format="wav")
            
            logging.info(f"Audio exportado exitosamente con pydub: {output_path}")

        except Exception as e:
            logging.error(f"Error en export_for_whisper: {e}")
            raise