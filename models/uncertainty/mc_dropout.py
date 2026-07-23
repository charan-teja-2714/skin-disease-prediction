import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from models.classification.model import EfficientNetClassifier

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "../classification/efficientnet_v5.pth"
IMAGE_PATH = "../../data/raw/images/ISIC_0024312.jpg"  # CHANGE THIS
IMG_SIZE = 224
N_PASSES = 30  # more passes → better uncertainty

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Using device:", device)

# --------------------------------------------------
# Enable Dropout (CRITICAL FIX)
# --------------------------------------------------
def enable_dropout(model):
    """
    Enable dropout layers during inference
    (Required for Monte Carlo Dropout)
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = EfficientNetClassifier(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

model.eval()            # keep BatchNorm stable
enable_dropout(model)  # force dropout ON

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError("❌ Image not found. Check IMAGE_PATH.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image / 255.0

image = torch.tensor(image, dtype=torch.float32)
image = image.permute(2, 0, 1).unsqueeze(0)
image = image.to(device)

# --------------------------------------------------
# MONTE CARLO DROPOUT INFERENCE
# --------------------------------------------------
predictions = []

with torch.no_grad():
    for _ in range(N_PASSES):
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        predictions.append(probs.cpu().numpy())

predictions = np.vstack(predictions)

# --------------------------------------------------
# UNCERTAINTY METRICS
# --------------------------------------------------
mean_probs = predictions.mean(axis=0)
std_probs = predictions.std(axis=0)

pred_class = np.argmax(mean_probs)
confidence = mean_probs[pred_class]
uncertainty = std_probs[pred_class]

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
print("\n📊 Prediction with Confidence & Uncertainty")
print("------------------------------------------------")
print(f"Predicted Class : {CLASS_NAMES[pred_class]}")
print(f"Confidence      : {confidence:.3f}")
print(f"Uncertainty     : {uncertainty:.3f}")

if uncertainty > 0.10:
    print("⚠️  Prediction UNCERTAIN – Dermatologist consultation recommended")
else:
    print("✅ Prediction CONFIDENT")
