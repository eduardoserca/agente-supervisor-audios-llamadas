"""
Script de prueba para verificar la diarización
"""
from agents.transcription_agent import TranscriptionAgent
import config

# Inicializar agente
print("Inicializando agente...")
agent = TranscriptionAgent(use_gpu=True)

# Probar con el primer archivo
test_file = "d:\\Proy\\AntiG\\MIA\\NLP\\TrabajoFinal\\output\\tmp\\temp_9155061852430001731_932711436_10053568_12_06.wav"

print(f"\nProbando diarización con: {test_file}")
result = agent.diarize_audio(test_file)

print(f"\nResultado de diarización:")
print(result)
