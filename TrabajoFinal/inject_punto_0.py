import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")
criteria_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\prompt\indicaciones_gestion_requerimiento.json")
analysis_dir = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\output\greeting_analysis")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new cells for Punto 0
punto_0_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 0. 📊 Generación de Data de Entrenamiento (Punto 0)\n",
        "En esta sección generamos el dataset para el fine-tuning combinando:\n",
        "1. **Reglas de Negocio**: Extraídas de `indicaciones_gestion_requerimiento.json`.\n",
        "2. **Análisis Reales**: Resultados previos de la carpeta `greeting_analysis`.\n",
        "3. **Escenarios Sintéticos**: Variaciones generadas para cubrir casos de cumplimiento e incumplimiento."
    ]
}

punto_0_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "import os\n",
        "import random\n",
        "from pathlib import Path\n",
        "\n",
        "# 🔧 Rutas de entrada y salida\n",
        "CRITERIA_FILE = Path(r\"./prompt/indicaciones_gestion_requerimiento.json\")\n",
        "ANALYSIS_DIR = Path(r\"./output/greeting_analysis\")\n",
        "OUTPUT_DIR = Path(r\"./data/pomptsft\")\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "DATASET_FILE = OUTPUT_DIR / \"dataset_audit.jsonl\"\n",
        "\n",
        "def generate_training_data():\n",
        "    print(\"🚀 Iniciando generación de data para Fine-tuning...\")\n",
        "    \n",
        "    # 1. Cargar Criterios\n",
        "    with open(CRITERIA_FILE, 'r', encoding='utf-8') as f:\n",
        "        criteria = json.load(f)\n",
        "    reglas = criteria.get(\"reglas\", {})\n",
        "    \n",
        "    dataset = []\n",
        "    \n",
        "    # 2. Procesar Archivos Reales de 'greeting_analysis'\n",
        "    if ANALYSIS_DIR.exists():\n",
        "        files = list(ANALYSIS_DIR.glob(\"*.json\"))\n",
        "        print(f\"📦 Encontrados {len(files)} archivos de análisis real.\")\n",
        "        \n",
        "        for file in files:\n",
        "            try:\n",
        "                with open(file, 'r', encoding='utf-8') as f:\n",
        "                    data = json.load(f)\n",
        "                \n",
        "                # Estructura esperada: transcription_text y rule_analysis\n",
        "                transcription = data.get(\"transcription_text\", \"\")\n",
        "                analysis = data.get(\"rule_analysis\", {})\n",
        "                \n",
        "                if transcription and analysis:\n",
        "                    # Crear par Alpaca\n",
        "                    dataset.append({\n",
        "                        \"instruction\": f\"Actúa como un auditor de calidad experto. Analiza la siguiente transcripción basándote en estas reglas: {list(reglas.keys())}. Responde estrictamente en JSON.\",\n",
        "                        \"input\": transcription,\n",
        "                        \"output\": json.dumps({\"rule_analysis\": analysis}, ensure_ascii=False)\n",
        "                    })\n",
        "            except Exception as e:\n",
        "                continue\n",
        "    \n",
        "    # 3. Generación de Escenarios Sintéticos (Validación de Escenarios)\n",
        "    print(\"🧪 Generando escenarios sintéticos adicionales...\")\n",
        "    \n",
        "    # Ejemplo de generador simple de variaciones\n",
        "    scenarios = [\n",
        "        {\"intent\": \"BAJA\", \"compliance\": True, \"text\": \"Cliente: Hola, quiero dar de baja mi servicio.\\nAsesor: Entiendo. ¿Me da su nombre, DNI, fecha de nacimiento y lugar? Además del monto.\\nCliente: Juan Perez, 12345678, 01/01/1980, Lima, monto 50 soles.\\nAsesor: Gracias, procedo con la baja.\", \"reason\": \"Validación completa realizada.\"},\n",
        "        {\"intent\": \"BAJA\", \"compliance\": False, \"text\": \"Cliente: Quiero la baja.\\nAsesor: Ok, deme su nombre y DNI.\\nCliente: Pedro Gomez, 87654321.\\nAsesor: Listo, ya está su baja.\", \"reason\": \"Faltó validación de fecha, lugar y monto para una baja.\"},\n",
        "        {\"intent\": \"CONSULTA\", \"compliance\": True, \"text\": \"Cliente: ¿Cual es mi saldo?\\nAsesor: Hola, su saldo es de 20 soles.\\nCliente: Gracias.\", \"reason\": \"Identificación básica suficiente para consulta.\"}\n",
        "    ]\n",
        "    \n",
        "    for sc in scenarios:\n",
        "        # Simular un JSON de salida basado en el escenario\n",
        "        fake_analysis = {\n",
        "            \"R1_validacion_datos\": {\"cumple\": sc[\"compliance\"], \"razon\": sc[\"reason\"], \"score\": 10 if sc[\"compliance\"] else 0}\n",
        "        }\n",
        "        dataset.append({\n",
        "            \"instruction\": \"Analiza la calidad del proceso de gestión del requerimiento en esta transcripción.\",\n",
        "            \"input\": sc[\"text\"],\n",
        "            \"output\": json.dumps({\"rule_analysis\": fake_analysis}, ensure_ascii=False)\n",
        "        })\n",
        "    \n",
        "    # 4. Guardar Dataset Final\n",
        "    random.shuffle(dataset)\n",
        "    with open(DATASET_FILE, 'w', encoding='utf-8') as f:\n",
        "        for entry in dataset:\n",
        "            f.write(json.dumps(entry, ensure_ascii=False) + \"\\n\")\n",
        "            \n",
        "    print(f\"✅dataset generado con {len(dataset)} ejemplos en: {DATASET_FILE}\")\n",
        "\n",
        "generate_training_data()"
    ]
}

# Rebuild the notebook with the new cells at the beginning
# We keep the original header (cell 0), then insert Punto 0 (cell 1 and 2)
new_cells = [nb['cells'][0], punto_0_md, punto_0_code] + nb['cells'][1:]

# Update indices in markdown headers if necessary (optional but nice)
# The previous 1. 2. 3. will stay as they are or we can re-index them
# Let's just re-index the markdown headers
idx = 1
for cell in new_cells:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        if source and source[0].startswith("## "):
            # If it already had a number, like "## 1. Environment", keep it or update it
            # For simplicity, we'll let them stay as they are, but Point 0 is correct now.
            pass

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully with Punto 0.")
