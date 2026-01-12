import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the indices of the relevant sections
# 1. Environment (## 1. 🛠️ Configuración del Entorno)
# 2. Model Loading (## 3. 🧠 Carga del Modelo Base)
# 3. Data Preparation (## 2. 📊 Preparación de Datos)

env_idx = -1
model_idx = -1
prep_idx = -1

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## 1. 🛠️ Configuración del Entorno" in source:
            env_idx = i
        elif "## 3. 🧠 Carga del Modelo Base" in source:
            model_idx = i
        elif "## 2. 📊 Preparación de Datos" in source:
            prep_idx = i

print(f"Indices found - Env: {env_idx}, Model: {model_idx}, Prep: {prep_idx}")

if env_idx != -1 and model_idx != -1 and prep_idx != -1:
    # We want: Punto 0 -> Env -> Model -> Prep -> Train -> Export
    # Current: Punto 0 -> Env -> Prep -> Model -> Train -> Export
    
    # Extract segments
    # 0. Header & Punto 0
    header_punto0 = cells[:env_idx]
    
    # 1. Environment (Markdown + Code)
    env_block = cells[env_idx:env_idx+2]
    
    # 2. Data Preparation (Markdown + Code)
    prep_block = cells[prep_idx:prep_idx+2]
    
    # 3. Model Loading (Markdown + Code)
    model_block = cells[model_idx:model_idx+2]
    
    # 4. Training (The rest)
    rest_block = cells[model_idx+2:]
    
    # Reassemble in correct order for Unsloth
    # New order: header_punto0 -> env_block -> model_block -> prep_block -> rest_block
    new_cells = header_punto0 + env_block + model_block + prep_block + rest_block
    
    # Update headers to be sequential
    for cell in new_cells:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            if source:
                line = source[0]
                if "## 1. " in line: pass
                elif "## 3. 🧠 Carga del Modelo Base" in line:
                    cell['source'][0] = line.replace("## 3. ", "## 2. ")
                elif "## 2. 📊 Preparación de Datos" in line:
                    cell['source'][0] = line.replace("## 2. ", "## 3. ")
    
    nb['cells'] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook reordered successfully. Model & Tokenizer now load before Data Preparation.")
else:
    print("Error: Could not find all section headers for reordering.")
