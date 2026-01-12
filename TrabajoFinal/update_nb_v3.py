
import json
import os

notebook_path = r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\analisis_audios_mejorado.ipynb"

# Detect indentation
with open(notebook_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    line2 = f.readline()
    indent = 0
    if line2:
        indent = len(line2) - len(line2.lstrip())

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
found_opt_b = False
for cell in nb['cells']:
    # Look for the cell that was Option B (Full Process)
    if cell['cell_type'] == 'code' and any("# OPCIÓN B: Transcripción + Diarización (Completo)" in line for line in cell['source']):
        found_opt_b = True
        
        # New Option B: Solo Diarización
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
        })
    elif cell['cell_type'] == 'markdown' and any("2. **Opción B (Transcripción + Diarización)**: Incluye" in line for line in cell['source']):
        # Update explanation for Option B
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.1 Procesar Audios\n",
                "\n",
                "Tienes dos opciones para procesar los audios:\n",
                "1. **Opción A (Solo Transcripción)**: Mucho más rápida, ideal si solo necesitas el texto.\n",
                "2. **Opción B (Solo Diarización)**: Añade la identificación de quién habla a los resultados de la Opción A. Es más lenta y requiere GPU."
            ]
        })
    else:
        new_cells.append(cell)

if not found_opt_b:
    print("Error: No se encontró la celda de la Opción B.")
else:
    nb['cells'] = new_cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=indent if indent > 0 else 4, ensure_ascii=False)
    print("Notebook actualizado exitosamente con la nueva Opción B.")
