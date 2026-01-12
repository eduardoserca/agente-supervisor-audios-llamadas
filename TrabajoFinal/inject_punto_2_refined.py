import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The refined convert_to_alpaca and loading logic (Punto 2)
refined_preparation_code = [
    "import json\n",
    "import os\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from datasets import load_dataset\n",
    "\n",
    "# 🔧 Configuración de rutas\n",
    "DATA_DIR = Path(\"./data/pomptsft\")\n",
    "DATASET_FILE = DATA_DIR / \"dataset_audit.jsonl\"\n",
    "ALPACA_TEMPLATE = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
    "\n",
    "### Instruction:\n",
    "{} \n",
    "\n",
    "### Input:\n",
    "{} \n",
    "\n",
    "### Response:\n",
    "{}\"\"\"\n",
    "\n",
    "def formatting_prompts_func(examples):\n",
    "    instructions = examples[\"instruction\"]\n",
    "    inputs       = examples[\"input\"]\n",
    "    outputs      = examples[\"output\"]\n",
    "    texts = []\n",
    "    for instruction, input_text, output in zip(instructions, inputs, outputs):\n",
    "        # Formatear cada ejemplo con el template de Alpaca\n",
    "        text = ALPACA_TEMPLATE.format(instruction, input_text, output) + tokenizer.eos_token\n",
    "        texts.append(text)\n",
    "    return { \"text\" : texts, }\n",
    "\n",
    "def load_and_prepare_dataset():\n",
    "    print(f\"📂 Cargando dataset desde: {DATASET_FILE}\")\n",
    "    \n",
    "    if not DATASET_FILE.exists():\n",
    "        print(\"❌ ERROR: El archivo de dataset no existe. Ejecuta el Punto 0 primero.\")\n",
    "        return None\n",
    "        \n",
    "    # Cargar usando la librería datasets de HuggingFace\n",
    "    dataset = load_dataset(\"json\", data_files=str(DATASET_FILE), split=\"train\")\n",
    "    \n",
    "    # Aplicar el formateo de Alpaca (mapeo para el entrenamiento)\n",
    "    dataset = dataset.map(formatting_prompts_func, batched = True,)\n",
    "    \n",
    "    print(f\"✅ Dataset cargado y formateado con {len(dataset)} ejemplos.\")\n",
    "    return dataset\n",
    "\n",
    "# Ejecutar la carga\n",
    "dataset = load_and_prepare_dataset()\n",
    "\n",
    "# Mostrar un ejemplo del texto final que verá el LLM\n",
    "if dataset:\n",
    "    print(\"\\n--- MUESTRA DEL FORMATO FINAL ---\")\n",
    "    print(dataset[0][\"text\"])\n"
]

# Find and replace Point 2 (Preparation)
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'convert_to_alpaca()' in "".join(cell['source']):
        cell['source'] = refined_preparation_code
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook updated with refined Punto 2 (Loading & Formatting).")
else:
    print("Error: Could not find the Punto 2 cell to update.")
