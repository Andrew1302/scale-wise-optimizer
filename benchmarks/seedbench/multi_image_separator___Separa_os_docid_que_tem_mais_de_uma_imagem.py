#!/usr/bin/env python3
import json
from pathlib import Path

# ==============================
# CONFIG
# ==============================

BASE_DIR = Path(".")
MULTI_IMAGE_FILE = "multi_image.jsonl"
OUTPUT_NAME = "val_filtered.jsonl"

# ==============================
# LOAD MULTI IMAGE IDS
# ==============================

def load_multi_image_ids(path):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["question_id"])
    return ids

# ==============================
# FILTER FILE
# ==============================

def filter_file(input_path, output_path, multi_ids):
    kept = 0
    removed = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)

            # pega question_id do seedbench
            sample_id = (
                data.get("seed_video", {}).get("question_id")
                or data.get("seed_image", {}).get("question_id")
                or data.get("seed_all", {}).get("question_id")
            )

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

    for res_dir in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()]):

        for model_dir in res_dir.glob("Qwen__*"):

            sample_files = list(model_dir.glob("*samples_seedbench.jsonl"))

            if not sample_files:
                continue

            input_file = sample_files[0]
            output_file = model_dir / OUTPUT_NAME

            print(f"Processing {res_dir.name}")

            kept, removed = filter_file(input_file, output_file, multi_ids)

            print(f"  kept {kept} | removed {removed}\n")

    print("Done.")

if __name__ == "__main__":
    main()