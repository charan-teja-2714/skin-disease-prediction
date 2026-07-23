import os
import sys
import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from models.classification.dataset import ISICClassificationDataset
from models.classification.model import EfficientNetClassifier

# ---------------- CONFIG ----------------
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "metadata.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_v5.pth")
BATCH_SIZE = 16
IMG_SIZE = 224

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- DATASET ----------------
dataset = ISICClassificationDataset(
    image_dir=IMAGE_DIR,
    csv_path=CSV_PATH,
    img_size=IMG_SIZE
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = EfficientNetClassifier(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------- METRICS ----------------
accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)

print("\nMODEL EVALUATION RESULTS")
print("----------------------------------")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)
