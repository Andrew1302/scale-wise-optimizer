#!/usr/bin/env python3
import json
import os
from pathlib import Path

# ==============================
# CONFIG
# ==============================

BASE_DIR = "."  # raiz onde estão 25k, 50k, etc
MULTI_IMAGE_FILE = "multi_image.txt"

OUTPUT_TEST = "test_filtered.jsonl"
OUTPUT_VAL = "val_filtered.jsonl"


# ==============================
# LOAD MULTI IMAGE IDS
# ==============================

def load_multi_image_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids


# ==============================
# FILTER FUNCTION
# ==============================

def filter_file(input_path, output_path, multi_ids):
    kept = 0
    removed = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)

            sample_id = data.get("mmmu_acc", {}).get("id")

            if sample_id in multi_ids:
                removed += 1
                continue

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

    return kept, removed


# ==============================
# MAIN
# ==============================

def main():
    multi_ids = load_multi_image_ids(MULTI_IMAGE_FILE)

    print(f"Loaded {len(multi_ids)} multi-image IDs\n")

    for res_dir in sorted(Path(BASE_DIR).iterdir()):
        if not res_dir.is_dir():
            continue

        # entra na subpasta do modelo
        model_dirs = list(res_dir.glob("Qwen__*"))
        if not model_dirs:
            continue

        model_dir = model_dirs[0]

        test_files = list(model_dir.glob("*samples_mmmu_test.jsonl"))
        val_files  = list(model_dir.glob("*samples_mmmu_val.jsonl"))

        if not test_files and not val_files:
            continue

        print(f"Processing {res_dir.name}...")

        # TEST
        if test_files:
            test_in = test_files[0]
            test_out = model_dir / OUTPUT_TEST

            kept, removed = filter_file(test_in, test_out, multi_ids)
            print(f"  TEST → kept {kept} | removed {removed}")

        # VAL
        if val_files:
            val_in = val_files[0]
            val_out = model_dir / OUTPUT_VAL

            kept, removed = filter_file(val_in, val_out, multi_ids)
            print(f"  VAL  → kept {kept} | removed {removed}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
