import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new combinatorial code for Punto 0
combinatorial_generator_code = [
    "import json\n",
    "import os\n",
    "import random\n",
    "from pathlib import Path\n",
    "\n",
    "# 🔧 Configuración\n",
    "CRITERIA_FILE = Path(r\"./prompt/indicaciones_gestion_requerimiento.json\")\n",
    "ANALYSIS_DIR = Path(r\"./output/greeting_analysis\")\n",
    "OUTPUT_DIR = Path(r\"./data/pomptsft\")\n",
    "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "DATASET_FILE = OUTPUT_DIR / \"dataset_audit.jsonl\"\n",
    "\n",
    "def generate_combinatorial_data(n_samples=1000):\n",
    "    \"\"\"Genera miles de ejemplos mezclando plantillas y estados de cumplimiento.\"\"\"\n",
    "    \n",
    "    names = [\"Juan Pérez\", \"María García\", \"Carlos Ruiz\", \"Ana Torres\", \"Luis Vega\", \"Elena Sol\"]\n",
    "    dnis = [\"11223344\", \"55667788\", \"99001122\", \"33445566\", \"77889900\"]\n",
    "    dates = [\"01/01/1980\", \"15/05/1992\", \"20/12/1975\", \"10/10/1988\"]\n",
    "    places = [\"Lima\", \"Cusco\", \"Arequipa\", \"Trujillo\", \"Piura\"]\n",
    "    amounts = [\"45.50\", \"89.90\", \"120.00\", \"35.00\", \"150.25\"]\n",
    "    \n",
    "    dataset = []\n",
    "    \n",
    "    print(f\"🚀 Generando {n_samples} ejemplos sintéticos combinatorios...\")\n",
    "    \n",
    "    for _ in range(n_samples):\n",
    "        # Elegir un escenario base al azar\n",
    "        scenario_type = random.choice([\"R1_BAJA\", \"R1_CONSULTA\", \"R3_RETENCION\", \"R6A_PREVIA\", \"R7_ESPERA\"])\n",
    "        name = random.choice(names)\n",
    "        dni = random.choice(dnis)\n",
    "        date = random.choice(dates)\n",
    "        place = random.choice(places)\n",
    "        amt = random.choice(amounts)\n",
    "        \n",
    "        if scenario_type == \"R1_BAJA\":\n",
    "            is_compliant = random.choice([True, False])\n",
    "            if is_compliant:\n",
    "                text = f\"Cliente: Quiero la baja.\\nAsesor: Por favor deme su nombre, DNI, fecha de nacimiento, lugar y último monto.\\nCliente: {name}, {dni}, {date}, {place} y mi monto fue {amt}.\\nAsesor: Gracias, todo validado.\"\n",
    "                analysis = {\"R1_validacion_datos\": {\"cumple\": True, \"razon\": \"Validación completa realizada.\", \"score\": 10}}\n",
    "            else:\n",
    "                text = f\"Cliente: Solicitó la cancelación del servicio.\\nAsesor: Deme su nombre y DNI.\\nCliente: {name}, {dni}.\\nAsesor: Perfecto, ya procedo.\"\n",
    "                analysis = {\"R1_validacion_datos\": {\"cumple\": False, \"razon\": \"Faltó pedir fecha, lugar y monto para una baja.\", \"score\": 0}}\n",
    "        \n",
    "        elif scenario_type == \"R1_CONSULTA\":\n",
    "            text = f\"Cliente: ¿Cual es mi deuda?\\nAsesor: Dígame su nombre y DNI.\\nCliente: {name}, {dni}.\\nAsesor: Su deuda es de {amt} soles.\"\n",
    "            analysis = {\"R1_validacion_datos\": {\"cumple\": True, \"razon\": \"Identificación básica correcta para consulta.\", \"score\": 10}}\n",
    "            \n",
    "        elif scenario_type == \"R3_RETENCION\":\n",
    "            n_offers = random.randint(1, 5)\n",
    "            turns = [f\"Cliente: Quiero la baja.\\nAsesor: Entiendo, valide con {name} y {dni}.\"]\n",
    "            for i in range(n_offers):\n",
    "                turns.append(f\"Asesor: ¿Y si le ofrecemos la oferta {i+1}?\\nCliente: No gracias.\")\n",
    "            \n",
    "            text = \"\\n\".join(turns)\n",
    "            compliant = n_offers <= 3\n",
    "            analysis = {\"R3_ofertas_adecuadas\": {\"cumple\": compliant, \"razon\": f\"Hizo {n_offers} ofertas.\", \"score\": 10 if compliant else 0}}\n",
    "            \n",
    "        elif scenario_type == \"R6A_PREVIA\":\n",
    "            is_compliant = random.choice([True, False])\n",
    "            if is_compliant:\n",
    "                text = f\"Cliente: Busco el código de mi baja de ayer.\\nAsesor: Claro {name}, el código es {random.randint(100,999)}. Estado: Ejecutado.\"\n",
    "                analysis = {\"R6A_consulta_baja_previa\": {\"cumple\": True, \"razon\": \"Información brindada.\", \"score\": 10}}\n",
    "            else:\n",
    "                text = f\"Cliente: ¿Estado de mi baja previa?\\nAsesor: No me aparece nada en sistema. Llame mañana.\"\n",
    "                analysis = {\"R6A_consulta_baja_previa\": {\"cumple\": False, \"razon\": \"No asistió al cliente con la información previa.\", \"score\": 0}}\n",
    "        \n",
    "        elif scenario_type == \"R7_ESPERA\":\n",
    "            wait_time = random.randint(1, 20)\n",
    "            text = f\"Asesor: Espere un momento...\\n({wait_time} minutos de espera)\\nCliente: ¿Hay alguien?\"\n",
    "            compliant = wait_time < 5\n",
    "            analysis = {\"R7_tiempo_espera_justificado\": {\"cumple\": compliant, \"razon\": f\"Espera de {wait_time} min.\", \"score\": 10 if compliant else 0}}\n",
    "            \n",
    "        dataset.append({\n",
    "            \"instruction\": \"Evalúa la calidad del servicio según las reglas de auditoría.\",\n",
    "            \"input\": text,\n",
    "            \"output\": json.dumps({\"rule_analysis\": analysis}, ensure_ascii=False)\n",
    "        })\n",
    "    \n",
    "    # Agregar data real si existe\n",
    "    if ANALYSIS_DIR.exists():\n",
    "        files = list(ANALYSIS_DIR.glob(\"*.json\"))\n",
    "        print(f\"📦 Integrando {len(files)} auditorías reales...\")\n",
    "        for file in files:\n",
    "            try:\n",
    "                with open(file, 'r', encoding='utf-8') as f:\n",
    "                    data = json.load(f)\n",
    "                if \"transcription_text\" in data and \"rule_analysis\" in data:\n",
    "                    dataset.append({\n",
    "                        \"instruction\": \"Evalúa la calidad del servicio según las reglas de auditoría.\",\n",
    "                        \"input\": data[\"transcription_text\"],\n",
    "                        \"output\": json.dumps({\"rule_analysis\": data[\"rule_analysis\"]}, ensure_ascii=False)\n",
    "                    })\n",
    "            except: continue\n",
    "\n",
    "    random.shuffle(dataset)\n",
    "    with open(DATASET_FILE, 'w', encoding='utf-8') as f:\n",
    "        for entry in dataset:\n",
    "            f.write(json.dumps(entry, ensure_ascii=False) + \"\\n\")\n",
    "            \n",
    "    print(f\"✅ DATASET CONSTRUIDO CON {len(dataset)} EJEMPLOS TOTALES.\")\n",
    "\n",
    "generate_combinatorial_data(n_samples=2000) # Generamos 2000 sintéticos + reales\n"
]

# Find the cell for Punto 0 and replace its source
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'generate_training_data()' in "".join(cell['source']):
        cell['source'] = combinatorial_generator_code
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook updated with combinatorial generator.")
else:
    print("Error updating notebook.")
