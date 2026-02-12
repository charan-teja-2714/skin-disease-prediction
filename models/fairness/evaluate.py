import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import cv2
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from models.fairness.skin_tone import estimate_skin_tone
from models.fairness.metrics import confusion_stats, false_negative_rate
from models.classification.model import EfficientNetClassifier


IMAGE_DIR = "../../data/raw/images"
CSV_PATH = "../../data/raw/metadata.csv"

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load EfficientNet classifier
model = EfficientNetClassifier(num_classes=7)
model.load_state_dict(torch.load(
    os.path.join(PROJECT_ROOT, "models", "classification", "efficientnet_best.pth"),
    map_location=device
))
model.to(device)
model.eval()

df = pd.read_csv(CSV_PATH)

group_preds = defaultdict(list)
group_labels = defaultdict(list)

for _, row in df.iterrows():
    image_id = row["image"]
    img_path = f"{IMAGE_DIR}/{image_id}.jpg"

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tone = estimate_skin_tone(image)

    label = None
    for i, cls in enumerate(CLASS_NAMES):
        if row[cls] == 1:
            label = i
            break

    if label is None:
        continue

    img = cv2.resize(image, (224, 224)) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(1).item()

    # Binary melanoma vs non-melanoma (clinical fairness)
    group_preds[tone].append(1 if pred == 0 else 0)
    group_labels[tone].append(1 if label == 0 else 0)

print("\nFairness Evaluation (Melanoma Detection) - EfficientNet")
for tone in group_preds:
    stats = confusion_stats(group_preds[tone], group_labels[tone])
    fnr = false_negative_rate(stats)
    print(f"{tone.capitalize()} skin -> FNR: {fnr:.3f}")
