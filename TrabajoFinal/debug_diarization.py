
import sys
import os
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import config

def debug_diarization():
    print("Loading pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.HUGGINGFACE_TOKEN
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    # Generate synthetic audio to avoid file issues
    print("Generating synthetic audio...")
    sr = 16000
    duration = 5 # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simple sine wave
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Ensure shape (channels, samples)
    if len(audio_data.shape) == 1:
        audio_data = audio_data[np.newaxis, :]
    
    waveform = torch.from_numpy(audio_data.astype(np.float32))
    sample_rate = sr
    print(f"Audio generated. Shape: {waveform.shape}, SR: {sample_rate}")

    print("Running diarization...")
    try:
        # Run pipeline
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=1,
            max_speakers=2
        )
        
        print("\n--- Diarization Result Info ---")
        print(f"Type: {type(diarization)}")
        
        # Generic extraction logic (equivalent to transcription_agent.py)
        annotation = None
        if hasattr(diarization, 'itertracks'):
            annotation = diarization
        elif hasattr(diarization, 'annotation'):
            annotation = diarization.annotation
        elif isinstance(diarization, dict) and 'annotation' in diarization:
            annotation = diarization['annotation']

        if annotation and hasattr(annotation, 'itertracks'):
            print("Successfully extracted annotation and 'itertracks' is available.")
            print(f"Sample tracks: {list(annotation.itertracks(yield_label=True))}")
        else:
            print("FAILED to extract valid annotation.")
            print(f"Dir: {dir(diarization)}")
            print(f"Str: {str(diarization)}")

    except Exception as e:
        print(f"Error running diarization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_diarization()
