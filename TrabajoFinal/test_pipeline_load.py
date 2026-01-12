"""
Script de prueba para verificar la carga del pipeline de diarización
"""
import torch
from pyannote.audio import Pipeline
import config

print("="*60)
print("PRUEBA DE CARGA DE PIPELINE DE DIARIZACIÓN")
print("="*60)

print(f"\nToken HF configurado: {config.HUGGINGFACE_TOKEN[:20]}...")
print(f"CUDA disponible: {torch.cuda.is_available()}")

try:
    print("\nCargando pipeline de diarización...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=config.HUGGINGFACE_TOKEN
    )
    print("✓ Pipeline cargado exitosamente")
    
    # Mover a GPU si está disponible
    if torch.cuda.is_available():
        print("Moviendo pipeline a GPU...")
        pipeline.to(torch.device("cuda"))
        print("✓ Pipeline en GPU")
    
    print(f"\nTipo de pipeline: {type(pipeline)}")
    print(f"Métodos disponibles: {[m for m in dir(pipeline) if not m.startswith('_')][:10]}")
    
except Exception as e:
    print(f"\n✗ ERROR cargando pipeline:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
