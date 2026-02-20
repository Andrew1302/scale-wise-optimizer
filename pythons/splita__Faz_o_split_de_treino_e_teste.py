#!/usr/bin/env python3
"""
splita.py
Divide dataset.jsonl → train/test
"""

import json
from sklearn.model_selection import train_test_split

# ╔════════ CONFIG ════════╗
INPUT_FILE = "dataset.jsonl"
TRAIN_OUT  = "train.jsonl"
TEST_OUT   = "test.jsonl"

TRAIN_FRAC = 0.8
SEED = 42
# ╚════════════════════════╝


def load_jsonl(path):

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def save_jsonl(data, path):

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():

    data = load_jsonl(INPUT_FILE)

    ids = [d["doc_id"] for d in data]

    train_ids, test_ids = train_test_split(
        ids,
        train_size=TRAIN_FRAC,
        random_state=SEED,
        shuffle=True
    )

    train = [d for d in data if d["doc_id"] in train_ids]
    test  = [d for d in data if d["doc_id"] in test_ids]

    save_jsonl(train, TRAIN_OUT)
    save_jsonl(test,  TEST_OUT)

    print("\nSaved:")
    print(f"{TRAIN_OUT} → {len(train)}")
    print(f"{TEST_OUT}  → {len(test)}")


if __name__ == "__main__":
    main()
