"""
Script de prueba completo para verificar la diarización
"""
from agents.transcription_agent import TranscriptionAgent
import config

# Inicializar agente
print("Inicializando agente...")
agent = TranscriptionAgent(use_gpu=True)

# Verificar que el pipeline se cargó
if agent.diarization_pipeline is None:
    print("\n✗ ERROR: Pipeline de diarización no se cargó")
    exit(1)
else:
    print("\n✓ Pipeline de diarización cargado correctamente")

# Probar con un archivo de audio original
test_file = r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\data\audios\RETENCIONES\9155061852430001731_932711436_10053568_12_06.wav"

print(f"\nProcesando audio completo: {test_file}")
result = agent.process_audio(test_file, preprocess=True)

print(f"\n{'='*60}")
print("RESULTADO DE PROCESAMIENTO")
print(f"{'='*60}")
print(f"Archivo: {result['audio_file']}")
print(f"Calidad: {result['metrics']['quality_score']:.1f}%")
print(f"Speakers detectados: {len(result['metrics']['speaker_metrics'])}")
print(f"Duración: {result['metrics']['overall_metrics']['audio_duration_seconds']:.1f}s")

# Mostrar primeros segmentos con speakers
print(f"\n{'='*60}")
print("PRIMEROS SEGMENTOS CON SPEAKERS")
print(f"{'='*60}")
for i, seg in enumerate(result['segments'][:10], 1):
    print(f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']} ({seg['role']}): {seg['text'][:50]}...")
