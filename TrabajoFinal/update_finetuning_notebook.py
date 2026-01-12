import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'OUTPUT_DATASET =' in source:
            cell['source'] = [
                "import json\n",
                "import os\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "# Configuración de rutas (Ajustar según tu proyecto)\n",
                "RESULTS_DIR = Path(\"./analysis_results/greeting\") # Donde están tus JSON\n",
                "DATA_DIR = Path(\"./data/pomptsft\")\n",
                "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
                "OUTPUT_DATASET = DATA_DIR / \"dataset_audit.jsonl\"\n",
                "\n",
                "def convert_to_alpaca():\n",
                "    dataset = []\n",
                "    \n",
                "    # Cargar todos los JSON de resultados\n",
                "    files = list(RESULTS_DIR.glob(\"*.json\"))\n",
                "    print(f\"Encontrados {len(files)} archivos de resultados.\")\n",
                "    \n",
                "    for file in files:\n",
                "        with open(file, 'r', encoding='utf-8') as f:\n",
                "            data = json.load(f)\n",
                "            \n",
                "            # Instrucción general\n",
                "            instruction = \"Evalúa la calidad de la siguiente transcripción de servicio al cliente según las reglas de auditoría. Responde en formato JSON.\"\n",
                "            \n",
                "            # Entrada: La transcripción\n",
                "            input_text = data.get(\"transcription_text\", \"\")\n",
                "            \n",
                "            # Salida: El análisis que el LLM hizo (y que tú validaste)\n",
                "            # Solo guardamos el rule_analysis para que el entrenamiento sea enfocado\n",
                "            output_json = {\"rule_analysis\": data.get(\"rule_analysis\", {})}\n",
                "            \n",
                "            if input_text and output_json[\"rule_analysis\"]:\n",
                "                dataset.append({\n",
                "                    \"instruction\": instruction,\n",
                "                    \"input\": input_text,\n",
                "                    \"output\": json.dumps(output_json, ensure_ascii=False)\n",
                "                })\n",
                "    \n",
                "    # Guardar en formato Alpaca JSON\n",
                "    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:\n",
                "        for entry in dataset:\n",
                "            f.write(json.dumps(entry, ensure_ascii=False) + \"\\n\")\n",
                "            \n",
                "    print(f\"Dataset creado con {len(dataset)} ejemplos: {OUTPUT_DATASET}\")\n",
                "\n",
                "convert_to_alpaca()"
            ]
            print("Updated data preparation cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully.")
