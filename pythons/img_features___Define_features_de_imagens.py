import os
import json
import numpy as np
import cv2
import pytesseract
from PIL import Image
from tqdm import tqdm

# ---------------------------------------
# CONFIG
# ---------------------------------------

DATASET_JSONL = "/media/vm03/ssd1T/ic/lmms-eval/train_seedbench/train.jsonl"
IMG_ROOT      = "/media/vm03/ssd1T/ic/lmms-eval/train_seedbench/images"

OUTPUT_PATH   = "/media/vm03/ssd1T/ic/lmms-eval/train_seedbench/img_features.jsonl"


# ---------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------

def extract_image_features(image_path):

    feats = {
        "aspect_ratio": 0,
        "area": 0,
        "edge_density": 0,
        "entropy": 0,
        "mean_letter_height": 0,
        "std_letter_height": 0,
        "ocr_char_count": 0
    }

    if not os.path.exists(image_path):
        return feats

    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # Resolution
        h, w = img_np.shape[:2]
        feats["aspect_ratio"] = w / h
        feats["area"] = int(w * h)

        # Edge density
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        feats["edge_density"] = float(edges.mean())

        # Entropy
        hist = np.histogram(gray, bins=256)[0]
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-8))
        feats["entropy"] = float(entropy)

        # OCR text
        ocr_text = pytesseract.image_to_string(img)
        feats["ocr_char_count"] = int(len(ocr_text))

        # OCR bounding boxes
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT
        )

        heights = [
            int(h) for h in data["height"]
            if str(h).isdigit() and int(h) > 0
        ]

        if heights:
            feats["mean_letter_height"] = float(np.mean(heights))
            feats["std_letter_height"]  = float(np.std(heights))

    except Exception as e:
        # Falha silenciosa → mantém zeros
        pass

    return feats


# ---------------------------------------
# LOAD DATASET
# ---------------------------------------

samples = []

with open(DATASET_JSONL) as f:
    for line in f:
        samples.append(json.loads(line))


# ---------------------------------------
# PROCESS
# ---------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as out_file:

    for sample in tqdm(samples, desc="Extracting image features"):

        # Ajusta conforme teu dataset
        doc_id = sample.get("id") or sample.get("doc_id")

        image_rel_path = sample.get("image")

        if image_rel_path:
            image_path = os.path.join(IMG_ROOT, image_rel_path)
        else:
            image_path = None

        feats = extract_image_features(image_path)

        record = {
            "doc_id": doc_id,
            **feats
        }

        out_file.write(json.dumps(record) + "\n")


print(f"\n✅ Features salvas em: {OUTPUT_PATH}")
