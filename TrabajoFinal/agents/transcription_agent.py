"""
Agente de transcripción de audio
Utiliza Whisper para transcripción y pyannote para diarización de speakers
"""
#
from faster_whisper import WhisperModel
import torch
from pyannote.audio import Pipeline
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os
import soundfile as sf
import numpy as np
import logging
import torchaudio
import torchaudio.transforms as T

# Add parent directory to path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config
from agents.audio_processor import AudioProcessor
from agents.metrics_calculator import MetricsCalculator


class TranscriptionAgent:
    """Agente para transcribir audios con identificación de speakers"""
    
    def __init__(self, 
                 whisper_model_name: str = config.WHISPER_MODEL,
                 use_gpu: bool = True):
        """
        Inicializa el agente con optimizaciones de memoria y carga robusta.
        """
        # 1. Configuración de Dispositivo y Logging
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        # 2. Carga Optimizada de Whisper
        # Usamos float16 si estamos en CUDA para duplicar la velocidad y ahorrar VRAM
        logging.info(f"Cargando Whisper '{whisper_model_name}' en {self.device}...")
        
        try:
            # download_root puede configurarse para evitar descargas repetidas
            self.whisper_model = WhisperModel(whisper_model_name, device=self.device,compute_type="float16")
                      
        except Exception as e:
            logging.error(f"Error crítico cargando Whisper: {e}")
            raise

        # 3. Pipeline de Diarización con Carga Perezosa (Lazy Loading)
        self.diarization_pipeline = None
        self._init_diarization()

        # 4. Inicialización de Componentes Auxiliares
        self.audio_processor = AudioProcessor()
        self.metrics_calculator = MetricsCalculator()
        self.whisper_prompt, self.whisper_hotwords = self._load_initial_prompt()

    def _log_system_info(self, use_gpu):
        """Maneja el diagnóstico del sistema de forma limpia."""
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        print(f"\n{'='*40}\nSISTEMA: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Dispositivo: {self.device.type.upper()}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif use_gpu:
            print("ADVERTENCIA: GPU no detectada. Usando CPU.")
        print(f"{'='*40}\n")

    def _init_diarization(self):
        """Carga el pipeline de Pyannote con validación de token."""
        token = getattr(config, 'HUGGINGFACE_TOKEN', None)
        if not token:
            logging.warning("Sin Token de HF: Diarización desactivada.")
            return

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token
            )
            if self.device.type == "cuda":
                # Pyannote requiere mover el pipeline completo
                #self.diarization_pipeline.to(self.device)
                self.diarization_pipeline.to(torch.device("cuda"))
            logging.info("Pipeline de diarización cargado correctamente.")
        except Exception as e:
            logging.error(f"Error cargando diarización: {e}")

    def _load_initial_prompt(self) -> tuple:
        """Carga el prompt de contexto y hotwords para Whisper de forma separada."""
        initial_prompt = ""
        hotwords = ""
        
        # 1. Cargar Prompt Inicial
        initial_path = getattr(config, "WHISPER_PROMPT_PATH", None)
        if initial_path and Path(initial_path).exists():
            try:
                initial_prompt = Path(initial_path).read_text(encoding="utf-8").strip()
            except Exception as e:
                logging.error(f"Error leyendo prompt inicial: {e}")
        
        # 2. Cargar Hotwords / Contexto Técnico
        hotwords_path = getattr(config, "WHISPER_HOTWORDS_PATH", None)
        if hotwords_path and Path(hotwords_path).exists():
            try:
                hotwords = Path(hotwords_path).read_text(encoding="utf-8").strip()
            except Exception as e:
                logging.error(f"Error leyendo hotwords context: {e}")
        
        # Valores por defecto si están vacíos
        if not initial_prompt and not hotwords:
            initial_prompt = "Conversación telefónica en español entre un agente y un cliente."
            
        return initial_prompt, hotwords
    
    def transcribe_audio(self, audio_path: str, language: str = "es") -> Dict:
        """
        Transcribe audio usando Whisper con parámetros optimizados para llamadas
        y prevención de alucinaciones.
        """
        logging.info(f"Iniciando transcripción: {audio_path}")
        
        try:
            # 1. Carga y Preparación (Aprovechando el preprocesamiento previo)
            # Se asume que el audio ya viene limpio del procesador
            audio_segment = self.audio_processor.load_audio(str(audio_path))
            
            # Whisper exige estrictamente 16000Hz y Mono
            if audio_segment.frame_rate != 16000:
                audio_segment = audio_segment.set_frame_rate(16000)
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
                
            # Convertir a float32 normalizado (Requerido por Whisper)
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            # Normalización basada en el ancho de bits (usualmente 16-bit para wav)
            max_int = float(2**(8 * audio_segment.sample_width - 1))
            audio_data = samples / max_int

            # 3. Ejecución de la transcripción

            segments_gen, info = self.whisper_model.transcribe(
                audio_data,
                language="es",             # Forzamos español para evitar que detecte inglés por error
                initial_prompt=self.whisper_prompt,
                hotwords=self.whisper_hotwords,
                
                # --- PARÁMETROS DE PRECISIÓN ---
                beam_size=5,               # Mayor precisión
                best_of=5,             # Genera varias opciones y elige la mejor
                patience=2.0,          # Búsqueda más exhaustiva de palabras
                # --- PREVENCIÓN DE ERRORES EN CADENA ---
                repetition_penalty=1.15,         # Penaliza palabras repetidas
                condition_on_previous_text=False, # Evita que un error se arrastre al siguiente tramo
                temperature=[0.0, 0.2, 0.4],
                # --- FILTROS DE ALUCINACIÓN ---
                compression_ratio_threshold=2.0, # Si el texto es muy repetitivo, lo marca como error
                log_prob_threshold=-1.0,         # Si la confianza es muy baja, reintenta
                no_speech_threshold=0.6,         # Umbral para descartar ruidos como si fueran voz
                
                # --- VAD Y DICCIONARIO ---
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Convertimos el generador a una lista para poder procesarlo
            segments = list(segments_gen)
            # 3. CONSTRUCCIÓN DEL DICCIONARIO (Para mantener compatibilidad con tu código)
            result = {
                "language": info.language,
                "duration": info.duration, # info ya trae la duración calculada
                "segments": []
            }
            # 4. Enriquecimiento de resultados
            result["duration"] = len(audio_segment) / 1000.0
            
            
            # Texto consolidado limpio
            #result["text"] = " ".join([s["text"].strip() for s in result["segments"]])
            # 4. Procesamiento de segmentos
            # En faster-whisper, los segmentos son OBJETOS, no diccionarios.
            # Se accede con .text y .no_speech_prob en lugar de ["text"]
            processed_segments = []
            for s in segments:
                if s.text.strip() and s.no_speech_prob < 0.8:
                    processed_segments.append({
                        "text": s.text.strip(),
                        "start": s.start,
                        "end": s.end,
                        "no_speech_prob": s.no_speech_prob,
                        "avg_logprob": s.avg_logprob,
                        "compression_ratio": s.compression_ratio
                    })
            
            result["segments"] = processed_segments
            result["text"] = " ".join([s["text"] for s in processed_segments])
            return result

        except Exception as e:
            logging.error(f"Fallo en transcripción de {audio_path}: {str(e)}")
            raise e
        finally:
            # Forzar limpieza de memoria para audios largos
            if 'samples' in locals(): del samples
            if 'audio_data' in locals(): del audio_data

    def diarize_audio_OLD(self, audio_path: str) -> Optional[Dict]:
        """
        Realiza diarización optimizada para velocidad.
        """
        if not self.diarization_pipeline:
            return None

        start_time = time.time()
        
        try:
            # 1. Carga optimizada: Leer solo lo necesario y en 16k
            # Usamos subseguentos o audios ya pre-procesados para ahorrar tiempo
            audio_data, sample_rate = sf.read(audio_path, dtype='float32')
            
            # 2. Downsampling a 16kHz si es mayor (la diarización es más rápida a 16k)
            if sample_rate > 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # 3. Preparación de Waveform (Directo a GPU)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1) # Convertir a Mono rápido
            
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            
            # Mover a dispositivo una sola vez
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            waveform = waveform.to(device)

            # 4. Inferencia con modo de alto rendimiento
            with torch.inference_mode():
                # El pipeline de pyannote acepta el tensor directamente
                diarization = self.diarization_pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    min_speakers=getattr(config, 'MIN_SPEAKERS', None), # Optimización: En llamadas siempre buscamos al menos 2
                    max_speakers=getattr(config, 'MAX_SPEAKERS', None)
                )

            # 5. Procesamiento de resultados
            # Intentamos obtener la anotación directamente (pyannote 3.0+)
            annotation = getattr(diarization, 'speaker_diarization', diarization)

            # List comprehension optimizada
            speaker_segments = [
                {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]

            # 6. Agrupación de segmentos (Opcional pero recomendado para velocidad de lectura posterior)
            # Esto une segmentos cortos del mismo speaker que están muy pegados
            optimized_segments = self._merge_adjacent_segments(speaker_segments)

            elapsed = time.time() - start_time
            print(f"Diarización veloz completada: {elapsed:.2f}s para {len(optimized_segments)} segmentos.")
            
            return {"segments": optimized_segments}

        except Exception as e:
            print(f"Error en diarización: {e}")
            return None
        finally:
            # No vaciamos cuda.empty_cache() aquí a menos que sea el final del proceso total
            # para evitar el overhead de reasignación.
            if 'waveform' in locals():
                del waveform



    def diarize_audio(self, audio_path: str) -> Optional[Dict]:
        if not self.diarization_pipeline:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_time = time.time()
        
        try:
            # 1. CARGA ULTRA-RÁPIDA con Torchaudio (Directo a Tensor)
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)

            # 2. RESAMPLING EN GPU (Mucho más rápido que librosa)
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000).to(device)
                waveform = resampler(waveform)
                sample_rate = 16000

            # 3. CONVERTIR A MONO EN GPU
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 4. INFERENCIA OPTIMIZADA
            # Nota: Asegúrate que el pipeline ya esté en GPU en el __init__: self.pipeline.to(device)
            with torch.inference_mode():
                diarization = self.diarization_pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    # TRUCO DE ORO: En llamadas, fijar num_speakers acelera el clustering un 40%
                    num_speakers=2 
                )

            # 5. EXTRACCIÓN DIRECTA
            # Usamos generadores para no saturar memoria
            annotation = getattr(diarization, 'speaker_diarization', diarization)
            segments = [
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker
                }
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]

            optimized_segments = self._merge_adjacent_segments(segments)
            
            elapsed = time.time() - start_time
            logging.info(f"Diarización completada en {elapsed:.2f}s")
            
            return {"segments": optimized_segments}

        except Exception as e:
            logging.error(f"Error en diarización: {e}")
            return None

    def _merge_adjacent_segments(self, segments, threshold=0.5):
        """Une segmentos del mismo speaker si el silencio entre ellos es menor al threshold."""
        if not segments: return []
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if next_seg['speaker'] == current['speaker'] and (next_seg['start'] - current['end']) < threshold:
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        return merged

    def merge_transcription_with_speakers(self, 
                                         transcription: Dict,
                                         diarization: Optional[Dict]) -> List[Dict]:
        """
        Combina transcripción con información de speakers
        
        Args:
            transcription: Resultado de Whisper
            diarization: Resultado de diarización
        
        Returns:
            Lista de segmentos con speaker asignado
        """
        segments = []
        
        for segment in transcription.get("segments", []):
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_text = segment["text"].strip()
            
            # Encontrar speaker correspondiente
            best_speaker = "SPEAKER_UNKNOWN"
            max_overlap = 0.0
            
            if diarization:
                for diar_seg in diarization["segments"]:
                    # Calcular overlap
                    overlap_start = max(seg_start, diar_seg["start"])
                    overlap_end = min(seg_end, diar_seg["end"])
                    overlap = overlap_end - overlap_start
                    
                    if overlap > 0:
                        # Si este speaker tiene mayor overlap que el anterior encontrado
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = diar_seg["speaker"]
            
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "speaker": best_speaker,
                "text": seg_text,
                "confidence": 1.0 - segment.get("no_speech_prob", 0.0), # Corregido: 1.0 - prob_no_voz
                "avg_logprob": segment.get("avg_logprob", 0.0),
                "compression_ratio": segment.get("compression_ratio", 0.0)
            })
        
        return segments
    
    def assign_roles(self, segments: List[Dict]) -> List[Dict]:
        """
        Asigna roles de 'AGENT' y 'CLIENT' basados en el contenido
        
        Args:
            segments: Lista de segmentos con speaker ya asignado
            
        Returns:
            Lista de segmentos con campo 'role' añadido
        """
        if not segments:
            return []
            
        # Agrupar texto por speaker
        speaker_texts = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(seg["text"])
            
        # Palabras clave de agente (desde configuración)
        agent_keywords = config.AGENT_KEYWORDS
        
        # Calcular score para cada speaker
        speaker_scores = {}
        for speaker, texts in speaker_texts.items():
            if speaker == "SPEAKER_UNKNOWN":
                speaker_scores[speaker] = -1
                continue
                
            score = 0
            # Unir todo el texto y convertir a mayúsculas para búsqueda insensible a mayúsculas/minúsculas
            full_text = " ".join(texts).upper()
            
            # Verificar keywords (convertidas también a mayúsculas)
            for keyword, weight in agent_keywords.items():
                if keyword.upper() in full_text:
                    # Contar ocurrencias
                    count = full_text.count(keyword.upper())
                    score += count * weight
            
            # Heurística posicional: El agente suele hablar en los primeros 3 segmentos
            first_segments = segments[:3]
            for i, seg in enumerate(first_segments):
                if seg["speaker"] == speaker:
                    # Dar puntos extra si aparece al inicio (más puntos si es el 1ro)
                    score += (3 - i)
            
            speaker_scores[speaker] = score
            
        # Identificar speaker con mayor score (Agente)
        agent_speaker = None
        if speaker_scores:
            agent_speaker = max(speaker_scores, key=speaker_scores.get)
            
        # Asignar roles
        for seg in segments:
            speaker = seg["speaker"]
            if speaker == "SPEAKER_UNKNOWN":
                seg["role"] = "UNKNOWN"
            elif speaker == agent_speaker:
                seg["role"] = "AGENT"
            else:
                seg["role"] = "CLIENT"
                
        return segments

    def process_audio(self, audio_path: str, 
                     preprocess: bool = True,
                     language: str = "es",
                     diarize: bool = True) -> Dict:
        """
        Procesa un archivo de audio completo
        
        Args:
            audio_path: Ruta al archivo de audio
            preprocess: Si se debe preprocesar el audio
            language: Idioma del audio
            diarize: Si se debe realizar diarización de speakers
        
        Returns:
            Diccionario con transcripción completa y métricas
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        # Preprocesar audio si se solicita
        preprocessing_metrics = None
        processed_audio_path = audio_path
        
        if preprocess:
            
            logging.info(f"Preprocesando audio...")
            # Forzar extension .wav para el archivo temporal para evitar errores con soundfile
            temp_path = config.TEMPORAL_DIR / f"temp_{audio_path.stem}.wav"
            audio = self.audio_processor.load_audio(str(audio_path))
            original_duration_ms = len(audio) # Guardar duración original


             # Aplicar filtro paso banda (voz humana)
            logging.info(f"Aplicando filtro paso banda...")
            audio = self.audio_processor.apply_bandpass_filter(audio)
            
            # Eliminar tonos de espera
            logging.info(f"Eliminando tonos de espera...")
            audio , hold_tones = self.audio_processor.remove_hold_tones(audio)
            
           
            # Eliminar silencios
            logging.info(f"Eliminando silencios...")
            audio, silence_pct = self.audio_processor.remove_silence(audio)
            
            # Exportar para Whisper
            logging.info(f"Exportando para Whisper...")
            self.audio_processor.export_for_whisper(audio, str(temp_path))
            
            preprocessing_metrics = {
                "silence_removed_percentage": silence_pct,
                "hold_tones_detected": len(hold_tones),
                "original_duration": original_duration_ms / 1000.0,
                "processed_duration": len(audio) / 1000.0
            }
            
            processed_audio_path = temp_path
        
        # Transcribir
        transcription = self.transcribe_audio(str(processed_audio_path), language)

        logging.info(f"Transcripción completada para {transcription}")
        
        # Diarizar si se solicita
        diarization = None
        if diarize:
            logging.info(f"Diarizando audio...")
            diarization = self.diarize_audio(str(processed_audio_path))
        else:
            logging.info(f"Diarización desactivada por el usuario.")
        
        # Combinar transcripción con speakers
        logging.info(f"Combinando transcripción con speakers...")        
        segments = self.merge_transcription_with_speakers(transcription, diarization)
        
        # Asignar roles (Agent vs Client)
        segments = self.assign_roles(segments)
        
        # Calcular duración del audio (preferir original si hubo preprocesamiento)
        if preprocessing_metrics and "original_duration" in preprocessing_metrics:
            audio_duration = preprocessing_metrics["original_duration"]
        else:
            audio_duration = transcription.get("duration", 0)
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        
        # Generar métricas
        metrics = self.metrics_calculator.generate_full_report(
            segments,
            processing_time,
            audio_duration,
            preprocessing_metrics
        )
        
        # Construir resultado
        result = {
            "audio_file": str(audio_path.name),
            "audio_path": str(audio_path.absolute()),
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "full_text": transcription.get("text", ""),
            "segments": segments,
            "metrics": metrics,
            "model_info": {
                "whisper_model": config.WHISPER_MODEL,
                "device": str(self.device),
                "diarization_enabled": diarization is not None
            }
        }
        
        return result

    def add_diarization_to_results(self, results: List[Dict]) -> List[Dict]:
        """
        Toma una lista de resultados de solo transcripción y les añade la información de locutores
        
        Args:
            results: Lista de diccionarios de resultados previos
            
        Returns:
            Lista de resultados actualizados con diarización
        """
        if not results:
            print("No hay resultados para procesar.")
            return []
            
        updated_results = []
        for result in results:
            try:
                # Obtener ruta del audio - Priorizar audio procesado en TEMPORAL_DIR
                audio_file_name = result.get("audio_file", "")
                audio_stem = Path(audio_file_name).stem
                processed_path = config.TEMPORAL_DIR / f"temp_{audio_stem}.wav"
                
                audio_path = None
                if processed_path.exists():
                    audio_path = str(processed_path)
                    print(f"Usando audio procesado encontrado en audioPro: {processed_path.name}")
                else:
                    # Si no hay procesado, intentar ruta original
                    original_path = result.get("audio_path")
                    if original_path and Path(original_path).exists():
                        audio_path = original_path
                    else:
                        # Buscar en el directorio de audios por defecto
                        potential_path = Path(config.AUDIO_DIR) / audio_file_name
                        if potential_path.exists():
                            audio_path = str(potential_path)

                if not audio_path:
                    print(f"No se encontró el archivo de audio para {audio_file_name}")
                    updated_results.append(result)
                    continue
                
                print(f"\n--- Iniciando diarización para: {result.get('audio_file')} ---")
                start_aug = time.time()
                
                # Realizar diarización
                diarization = self.diarize_audio(audio_path)
                
                if not diarization:
                    print(f"Saltando diarización para {result.get('audio_file')} (falla o no disponible)")
                    updated_results.append(result)
                    continue
                
                # Re-combinar con los segmentos existentes
                # merge_transcription_with_speakers espera un dict con "segments"
                transcription_dummy = {"segments": result["segments"]}
                segments = self.merge_transcription_with_speakers(transcription_dummy, diarization)
                
                # Re-asignar roles
                segments = self.assign_roles(segments)
                
                # Obtener métricas previas para recalculado
                audio_duration = result["metrics"]["overall_metrics"]["audio_duration_seconds"]
                preprocessing_metrics = result["metrics"].get("preprocessing_metrics")
                
                # Tiempo de procesamiento acumulado
                aug_time = time.time() - start_aug
                prev_time = result["metrics"]["overall_metrics"]["processing_time_seconds"]
                
                # Generar reporte completo actualizado
                metrics = self.metrics_calculator.generate_full_report(
                    segments,
                    prev_time + aug_time,
                    audio_duration,
                    preprocessing_metrics
                )
                
                # Actualizar el resultado
                result["segments"] = segments
                result["metrics"] = metrics
                result["model_info"]["diarization_enabled"] = True
                
                # Guardar el JSON actualizado
                output_dir = Path(config.TRANSCRIPTION_DIR)
                output_file = output_dir / f"{Path(result['audio_file']).stem}_transcription.json"
                self.save_transcription(result, str(output_file))
                
                updated_results.append(result)
                print(f"✓ Diarización completada para {result.get('audio_file')}")
                
            except Exception as e:
                print(f"Error procesando diarización diferida para {result.get('audio_file')}: {e}")
                updated_results.append(result)
                
        return updated_results
    
    def save_transcription(self, result: Dict, output_path: str):
        """
        Guarda el resultado de transcripción en archivo JSON
        
        Args:
            result: Diccionario con resultado de transcripción
            output_path: Ruta del archivo de salida
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Transcripción guardada en: {output_path}")
    
    def process_directory(self, audio_dir: str = None, 
                         output_dir: str = None,
                         preprocess: bool = True,
                         diarize: bool = True) -> List[Dict]:
        """
        Procesa todos los audios en un directorio
        
        Args:
            audio_dir: Directorio con audios (por defecto config.AUDIO_DIR)
            output_dir: Directorio de salida (por defecto config.TRANSCRIPTION_DIR)
            preprocess: Si se debe preprocesar los audios
            diarize: Si se debe realizar diarización de speakers
        
        Returns:
            Lista de resultados de transcripción
        """
        audio_dir = Path(audio_dir or config.AUDIO_DIR)
        output_dir = Path(output_dir or config.TRANSCRIPTION_DIR)
        
        # Buscar archivos de audio
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"No se encontraron archivos de audio en {audio_dir}")
            return []
        
        print(f"Encontrados {len(audio_files)} archivos de audio")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n{'='*60}")
            print(f"Procesando {i}/{len(audio_files)}: {audio_file.name}")
            print(f"{'='*60}")
            
            try:
                # Procesar audio
                result = self.process_audio(str(audio_file), preprocess=preprocess, diarize=diarize)
               
                # Guardar resultado
                output_file = output_dir / f"{audio_file.stem}_transcription.json"
                print(f"Guardando transcripción: {output_file}")
                self.save_transcription(result, str(output_file))
                
                results.append(result)
                
                print(f"Completado: {audio_file.name}")
                print(f"  - Calidad: {result['metrics']['quality_score']:.1f}%")
                print(f"  - Speakers: {len(result['metrics']['speaker_metrics'])}")
                print(f"  - Duracion: {result['metrics']['overall_metrics']['audio_duration_seconds']:.1f}s")
                
            except Exception as e:
                print(f"Error procesando {audio_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"Procesamiento completado: {len(results)}/{len(audio_files)} exitosos")
        print(f"{'='*60}")
        
        return results


#if __name__ == "__main__":
    # Ejemplo de uso
    #agent = TranscriptionAgent()
    #results = agent.process_directory()
