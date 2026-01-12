import json
from pathlib import Path

notebook_path = Path(r"d:\Proy\AntiG\MIA\NLP\TrabajoFinal\finetuning_llm.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated_synthetic = False
updated_alpaca = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        source_text = "".join(source_lines)
        
        # Target Synthetic Data Generation Cell
        if 'output_path = Path("transcripciones_reglas.jsonl")' in source_text:
            new_source = []
            for line in source_lines:
                if 'output_path = Path("transcripciones_reglas.jsonl")' in line:
                    new_source.append('DATA_DIR = Path("./data/pomptsft")\n')
                    new_source.append('DATA_DIR.mkdir(parents=True, exist_ok=True)\n')
                    new_source.append('output_path = DATA_DIR / "transcripciones_reglas.jsonl"\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            updated_synthetic = True
            print("Updated synthetic data generation cell.")

        # Target Alpaca Conversion Cell
        if 'OUTPUT_DATASET = "dataset_audit.jsonl"' in source_text:
            new_source = []
            for line in source_lines:
                if 'OUTPUT_DATASET = "dataset_audit.jsonl"' in line:
                    # Check if DATA_DIR is already defined in this cell or use it if not
                    if 'DATA_DIR =' not in source_text:
                        new_source.append('DATA_DIR = Path("./data/pomptsft")\n')
                        new_source.append('DATA_DIR.mkdir(parents=True, exist_ok=True)\n')
                    new_source.append('OUTPUT_DATASET = DATA_DIR / "dataset_audit.jsonl"\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            updated_alpaca = True
            print("Updated Alpaca conversion cell.")

if updated_synthetic or updated_alpaca:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Notebook saved successfully.")
else:
    print("Could not find matching code patterns to update.")
