import json
import pandas as pd

# ====== INPUT ======
input_file = "input.jsonl"   # seu arquivo jsonl
output_file = "output.xlsx" # xlsx final

# ====== LOAD JSONL ======
rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# ====== DATAFRAME ======
df = pd.DataFrame(rows)

# Ordena colunas (opcional)
cols = ["doc_id", "target"] + sorted(
    [c for c in df.columns if c not in ["doc_id", "target"]],
    key=lambda x: int(x)
)
df = df[cols]

# ====== SAVE XLSX ======
df.to_excel(output_file, index=False)

print(f"Salvo em: {output_file}")
