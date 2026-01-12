
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from scipy import signal
from pathlib import Path
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config

def apply_bandpass_filter_debug(audio: AudioSegment, 
                        low_cut: int = 300, 
                        high_cut: int = 3400, 
                        order: int = 5) -> AudioSegment:
        # 1. Normalización inicial
        print(f"Original dBFS: {audio.dBFS:.2f}")
        # audio = effects.normalize(audio) # Probar sin esto si hay distorsión
        
        fs = audio.frame_rate
        nyquist = 0.5 * fs
        low = max(0.001, low_cut / nyquist)
        high = min(0.999, high_cut / nyquist)
        sos = signal.butter(order, [low, high], btype='band', output='sos')

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Guardar escala original para restaurar volumen
        original_max = np.max(np.abs(samples))
        print(f"Max original sample: {original_max}")

        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            filtered = np.zeros_like(samples)
            for i in range(audio.channels):
                filtered[:, i] = signal.sosfiltfilt(sos, samples[:, i])
        else:
            filtered = signal.sosfiltfilt(sos, samples)

        # DEBUG: Check for clipping before clipping
        filtered_max = np.max(np.abs(filtered))
        print(f"Max filtered sample: {filtered_max}")

        # 2. HARD GATE
        noise_threshold = np.max(np.abs(filtered)) * 0.05 
        print(f"Noise threshold: {noise_threshold}")
        filtered[np.abs(filtered) < noise_threshold] = 0 

        # 3. CLIP AND SPAWN
        max_val = float(2**(8 * audio.sample_width - 1))
        print(f"Max theoretical val: {max_val}")
        
        # Si filtered_max es mucho más alto que max_val, hay distorsión por overflow antes de clip
        
        clipped = np.clip(filtered, -max_val, max_val - 1).astype(np.int16)
        processed_audio = audio._spawn(clipped.tobytes())
        
        return processed_audio

def test_debug():
    audio_path = r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\output\audioPro\temp_9155033267980001731 999799232 70942859 MASIVO_MOVIL ABAI 0906 _1.wav"
    if not os.path.exists(audio_path):
        print(f"No se encontró el audio: {audio_path}")
        return

    print("Cargando audio...")
    audio = AudioSegment.from_file(audio_path)
    
    print("\n--- Ejecutando Filtro Debug ---")
    processed = apply_bandpass_filter_debug(audio)
    
    output_path = "debug_filter_output.wav"
    processed.export(output_path, format="wav")
    print(f"\nResultado guardado en: {output_path}")

if __name__ == "__main__":
    test_debug()
