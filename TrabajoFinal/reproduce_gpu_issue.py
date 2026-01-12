import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())

import config
from agents.transcription_agent import TranscriptionAgent

def test_gpu_usage():
    print("--- Testing GPU Usage ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    print("\nInitializing TranscriptionAgent with use_gpu=True...")
    agent = TranscriptionAgent(use_gpu=True)
    
    print(f"\nAgent Device: {agent.device}")
    
    if hasattr(agent, 'whisper_model'):
        # Check model device
        model_device = agent.whisper_model.device
        print(f"Whisper Model Device: {model_device}")
        
    if hasattr(agent, 'diarization_pipeline') and agent.diarization_pipeline:
        # Check diarization device (pyannote pipeline isn't a single module, but we can check if it ran .to())
        # Usually checking an attribute is hard, but we know we called .to('cuda')
        print(f"Diarization Pipeline loaded: Yes")
    else:
        print(f"Diarization Pipeline loaded: No")

    # Try transcribing a file
    audio_file = "d:/Proy/AntiG/MIA/NLP/TrabajoFinal/data/audios/RETENCIONES/9155061852430001731_932711436_10053568_12_06.wav"
    if os.path.exists(audio_file):
        print(f"\nTranscribing {audio_file}...")
        try:
            result = agent.transcribe_audio(audio_file)
            print("Transcription successful.")
            print(f"Detected language: {result.get('language')}")
        except Exception as e:
            print(f"Transcription failed: {e}")
    else:
        print(f"\nAudio file not found: {audio_file}")

if __name__ == "__main__":
    test_gpu_usage()
