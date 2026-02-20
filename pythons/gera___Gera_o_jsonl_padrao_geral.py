#!/usr/bin/env python3
"""
gera.py
Agrega scores multi-resolução → dataset.jsonl
"""

from pathlib import Path
import json

# ╔════════ CONFIG ════════╗
BASE_DIR = Path(".")
VAL_FILE = "val_filtered.jsonl"
OUT_FILE = "dataset.jsonl"
# ╚════════════════════════╝


def parse_resolution(res: str) -> str:
    s = str(res).lower().strip()
    if s.endswith("k"):
        return str(int(s[:-1]) * 1000)
    return str(int(s))


def compute_score(sample):

    acc = sample.get("mmmu_acc")
    if not acc:
        return None

    gt   = acc.get("answer")
    pred = acc.get("parsed_pred")

    if not gt or not pred:
        return None

    return int(str(gt).lower() == str(pred[0]).lower())


def main():

    samples = {}

    for res_dir in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()]):

        try:
            resolution = parse_resolution(res_dir.name)
        except:
            continue

        for val_path in res_dir.rglob(VAL_FILE):

            print(f"Processing {resolution} → {val_path}")

            with open(val_path, encoding="utf-8") as f:

                for line in f:

                    s = json.loads(line)

                    doc_id = s.get("doc_id")
                    text   = s.get("input")

                    if not doc_id or not text:
                        continue

                    score = compute_score(s)
                    if score is None:
                        continue

                    if doc_id not in samples:
                        samples[doc_id] = {
                            "doc_id": doc_id,
                            "input": text,
                            "scores": {}
                        }

                    samples[doc_id]["scores"][resolution] = score

    data = list(samples.values())

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {OUT_FILE} → {len(data)} samples")


if __name__ == "__main__":
    main()
