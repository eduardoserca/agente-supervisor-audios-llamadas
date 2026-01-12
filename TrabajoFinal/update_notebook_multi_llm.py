import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\analisis_audios_mejorado_MUltiagentes.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

init_cell_idx = -1
process_cell_idx = -1
visual_cell_exists = False

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if cell['cell_type'] == 'code':
        if 'MODELS_TO_COMPARE =' in source:
            init_cell_idx = i
        if 'process_directory(' in source and 'for model in MODELS_TO_COMPARE:' in source:
            process_cell_idx = i
        if 'comparison_data = []' in source and 'all_results.items()' in source:
            visual_cell_exists = True

# 1. Update Init Cell with MODELS definition (as dicts with URLs)
if init_cell_idx != -1:
    nb['cells'][init_cell_idx]['source'] = [
        "# Definir modelos y sus URLs a comparar\n",
        "MODELS_TO_COMPARE = [\n",
        "    {\n",
        "        \"model\": config.GPT_MODEL, \n",
        "        \"url\": config.OPENAI_BASE_URL, \n",
        "        \"name\": \"GPT-Local\"\n",
        "    },\n",
        "    {\n",
        "        \"model\": \"lmstudio-community/Meta-Llama-3-8B-Instruct\", \n",
        "        \"url\": \"http://localhost:1234/v1\", \n",
        "        \"name\": \"Llama-Local\"\n",
        "    },\n",
        "    # Puedes agregar más modelos aquí, incluso de APIs remotas\n",
        "    # {\n",
        "    #     \"model\": \"gpt-4o\", \n",
        "    #     \"url\": \"https://api.openai.com/v1\", \n",
        "    #     \"name\": \"OpenAI-Cloud\"\n",
        "    # }\n",
        "]\n",
        "\n",
        "all_results = {}\n",
        "print(f\"✓ {len(MODELS_TO_COMPARE)} configuraciones de modelos listas para comparación\")"
    ]
    print(f"Updated initialization cell at index {init_cell_idx}")

# 2. Update Process Cell with the loop handling dict entries
if process_cell_idx != -1:
    nb['cells'][process_cell_idx]['source'] = [
        "print(f\"Iniciando análisis comparativo con {len(MODELS_TO_COMPARE)} configuraciones de LLM\")\n",
        "\n",
        "for config_item in MODELS_TO_COMPARE:\n",
        "    model = config_item[\"model\"]\n",
        "    url = config_item[\"url\"]\n",
        "    display_name = config_item[\"name\"]\n",
        "    \n",
        "    print(f\"\\n{'#'*60}\")\n",
        "    print(f\"PROCESANDO CON: {display_name} ({model})\")\n",
        "    print(f\"URL: {url}\")\n",
        "    print(f\"{'#'*60}\")\n",
        "    \n",
        "    # Inicializar agente con el modelo y URL específicos\n",
        "    agent = QualityAgent(\n",
        "        criteria_path=str(config.CRITERIOS_JSON),\n",
        "        use_ai=True,\n",
        "        model_name=model,\n",
        "        base_url=url\n",
        "    )\n",
        "    \n",
        "    results = agent.process_directory(\n",
        "        transcription_dir=str(config.TRANSCRIPTION_DIR),\n",
        "        output_dir=str(config.GREETING_DIR)\n",
        "    )\n",
        "    # Guardar resultados usando el nombre amigable o el modelo como clave\n",
        "    all_results[display_name] = results\n",
        "\n",
        "print(\"\\n✓ Procesamiento multi-modelo y multi-URL completado\")"
    ]
    print(f"Updated processing cell at index {process_cell_idx}")

# 3. Update comparison data extraction to use the new keys
if visual_cell_exists:
    for cell in nb['cells']:
        source = "".join(cell['source'])
        if 'comparison_data = []' in source and 'all_results.items()' in source:
             cell['source'] = [
                "import pandas as pd\n",
                "import plotly.express as px\n",
                "\n",
                "# 1. Extraer datos para comparación\n",
                "comparison_data = []\n",
                "for config_name, results in all_results.items():\n",
                "    for res in results:\n",
                "        comparison_data.append({\n",
                "            \"Config\": config_name,\n",
                "            \"Audio\": res['audio_file'][:30], # Truncar para visualización\n",
                "            \"Score\": res['global_score'],\n",
                "            \"Compliance\": res['compliance']['global_result']\n",
                "        })\n",
                "\n",
                "if comparison_data:\n",
                "    df_comp = pd.DataFrame(comparison_data)\n",
                "\n",
                "    # 2. Gráfico de Barras Comparativo\n",
                "    fig = px.bar(df_comp, x=\"Audio\", y=\"Score\", color=\"Config\", barmode=\"group\",\n",
                "                 title=\"Comparación de Scores Globales por Configuración de LLM\",\n",
                "                 labels={\"Score\": \"Puntuación (0-100)\", \"Audio\": \"Archivo de Audio\", \"Config\": \"Configuración\"})\n",
                "    fig.show()\n",
                "\n",
                "    # 3. Resumen Promedio por Configuración\n",
                "    avg_scores = df_comp.groupby(\"Config\")[\"Score\"].mean().reset_index()\n",
                "    fig_avg = px.pie(avg_scores, values=\"Score\", names=\"Config\", \n",
                "                     title=\"Distribución de Calidad Promedio por Configuración\")\n",
                "    fig_avg.show()\n",
                "\n",
                "    print(\"Resumen Comparativo:\")\n",
                "    display(avg_scores)\n",
                "else:\n",
                "    print(\"⚠️ No hay datos para comparar\")"
             ]
             print("Updated visualization cell source")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully with Multi-URL support")
