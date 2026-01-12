"""
Script de prueba rápida (10 segundos) para verificar la diarización
"""
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
import config

print("Inicializando pipeline de diarización...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=config.HUGGINGFACE_TOKEN
)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# Archivo de prueba (TEMP)
test_file = r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\output\tmp\temp_9155061852430001731_932711436_10053568_12_06.wav"

# Leer solo 10 segundos (160000 frames at 16000Hz)
print(f"Leyendo 20 segundos de: {test_file}")
audio_data, sample_rate = sf.read(test_file, frames=20 * 16000)

# Asegurar forma (channels, time)
if len(audio_data.shape) == 1:
    audio_data = audio_data[np.newaxis, :]
else:
    audio_data = audio_data.T
    
# Convertir a tensor
waveform = torch.from_numpy(audio_data.astype(np.float32))
if torch.cuda.is_available():
    waveform = waveform.to(torch.device("cuda"))

print("Ejecutando diarización...")
diarization = pipeline(
    {"waveform": waveform, "sample_rate": sample_rate},
    min_speakers=1,
    max_speakers=2
)

print(f"Tipo de resultado: {type(diarization)}")
speaker_segments = []
# LA PRUEBA REAL: ver si itertracks funciona
for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

print(f"Se encontraron {len(speaker_segments)} segmentos:")
for seg in speaker_segments:
    print(seg)
