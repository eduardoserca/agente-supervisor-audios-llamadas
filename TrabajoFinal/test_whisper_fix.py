
import sys
import os
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from agents.transcription_agent import TranscriptionAgent
import config

def test_optimization():
    # Path to one of the audios in audioPro
    audio_pro_dir = Path(config.OUTPUT_DIR) / "audioPro"
    audio_files = list(audio_pro_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No se encontraron archivos en {audio_pro_dir}")
        return
    
    test_audio = audio_files[0]
    print(f"Probando transcripción tras corrección para: {test_audio.name}")
    
    agent = TranscriptionAgent()
    
    try:
        result = agent.transcribe_audio(str(test_audio))
        print("\n--- Resultado de Transcripción ---")
        print(f"Texto: {result.get('text', '')[:500]}...")
        print(f"Duración: {result.get('duration', 0):.2f}s")
        print(f"Segmentos detectados: {len(result.get('segments', []))}")
        print("--------------------------------")
        
        # Guardar resultado de prueba
        output_file = Path(current_dir) / "test_fix_result.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Archivo: {test_audio.name}\n")
            f.write(f"Texto: {result.get('text', '')}\n")
        
        print(f"Resultado guardado en: {output_file}")
        
    except Exception as e:
        print(f"Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimization()
