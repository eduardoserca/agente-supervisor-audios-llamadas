import numpy as np
import librosa
from scipy.ndimage import binary_closing
from typing import Tuple, List

def simulate_detect_hold_tones(sr=16000, duration=30, tone_segments=[], voice_segments=[(2, 5)], silence_ratio=0.8):
    """
    Simula la detección de tonos con datos sintéticos.
    """
    hop_length = 512
    n_fft = 2048
    n_frames = int(duration * sr / hop_length)
    
    # Simular energía en el rango 400-480Hz
    target_energy = np.zeros(n_frames)
    
    # Añadir un poco de voz (poca duración comparado con el total)
    for start, end in voice_segments:
        s_idx = int(start * sr / hop_length)
        e_idx = int(end * sr / hop_length)
        target_energy[s_idx:e_idx] = 0.5 
        
    # Ruido de fondo muy bajo
    target_energy += np.random.normal(0, 0.01, n_frames)
    target_energy = np.maximum(target_energy, 0)

    # Lógica nueva:
    # 1. Ratio de energía
    total_energy = target_energy + 0.1 # Simular energía fuera del rango (voz/ruido)
    energy_ratio = target_energy / (total_energy + 1e-6)
    
    # 2. Umbral absoluto
    energy_floor = np.percentile(total_energy, 20) * 2.0
    
    hold_tone_frames = (energy_ratio > 0.4) & (total_energy > energy_floor)
    
    gap_tolerance = 2.0
    gap_frames = int(gap_tolerance * sr / hop_length)
    refined_frames = binary_closing(hold_tone_frames, structure=np.ones(gap_frames))
    
    return hold_tone_frames, refined_frames, target_energy, energy_floor

if __name__ == "__main__":
    hold, refined, energy, thresh = simulate_detect_hold_tones()
    
    print(f"Total frames: {len(hold)}")
    print(f"Frames flagged as tone (raw): {np.sum(hold)}")
    print(f"Frames flagged as tone (refined): {np.sum(refined)}")
    
    if np.sum(refined) > len(hold) * 0.8:
        print("ALERTA: Se está detectando casi todo el audio como tono!")
