
import json
import os

notebook_path = r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\analisis_audios_mejorado.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    source_str = "".join(cell['source'])
    
    # Check for Markdown cell to update
    if cell['cell_type'] == 'markdown' and "2. **Opción B (Transcripción + Diarización)**" in source_str:
        print("Updating Markdown cell...")
        cell['source'] = [
            "### 3.1 Procesar Audios\n",
            "\n",
            "Tienes dos opciones para procesar los audios:\n",
            "1. **Opción A (Solo Transcripción)**: Mucho más rápida, ideal si solo necesitas el texto.\n",
            "2. **Opción B (Solo Diarización)**: Añade la identificación de quién habla a los resultados de la Opción A. Es más lenta y requiere GPU."
        ]
    
    # Check for Code cell to update
    if cell['cell_type'] == 'code' and "# OPCIÓN B: Transcripción + Diarización (Completo)" in source_str:
        print("Updating Code cell...")
        cell['source'] = [
            "# OPCIÓN B: Solo Diarización (Diferida)\n",
            "print(\"Iniciando SOLO DIARIZACIÓN basada en transcripción previa...\\n\")\n",
            "if 'transcription_results' in locals() and transcription_results:\n",
            "    transcription_results = transcription_agent.add_diarization_to_results(\n",
            "        results=transcription_results\n",
            "    )\n",
            "    print(f\"\\n✓ Diarización completada para {len(transcription_results)} audios\")\n",
            "else:\n",
            "    print(\"⚠️ Error: No se encontraron resultados de transcripción. Por favor, ejecuta la OPCIÓN A primero.\")"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Notebook actualizado.")
