"""
Professional ISIC 2018 training script.

Architecture : EfficientNet-B3  (10.7M params — right capacity for 10k images)
Loss         : Focal Loss (gamma=2) — invented for extreme class imbalance in medical imaging.
               Down-weights easy NV examples automatically (pt high → loss ≈ 0),
               forces gradient updates from hard minority-class examples.
               Eliminates the sampler+weighted-loss double-correction problem entirely.
Training     : Two-phase fine-tuning
               Phase 1 (epochs 1-8)  : backbone frozen, train head only at high LR
               Phase 2 (epochs 9-30) : full model, cosine LR decay from 1e-4 → 1e-6
Precision    : Mixed precision (AMP) — 2x faster, larger effective batch size
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from models.classification.dataset import ISICClassificationDataset
from models.classification.model import EfficientNetClassifier

# --------------- Config ---------------
IMAGE_DIR  = os.path.join(PROJECT_ROOT, "data/raw/images")
CSV_PATH   = os.path.join(PROJECT_ROOT, "data/raw/metadata.csv")
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "efficientnet_v3.pth")
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

PHASE1_EPOCHS = 8    # head-only warmup
PHASE2_EPOCHS = 22   # full fine-tune  (total = 30)
BATCH_SIZE    = 32
FOCAL_GAMMA   = 2.0  # standard value from RetinaNet paper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --------------- Focal Loss ---------------
class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al., 2017 (RetinaNet).
    FL(pt) = -(1 - pt)^gamma * log(pt)

    When the model is confident and correct (pt → 1), loss → 0.
    NV is easy to classify with high confidence → contributes almost nothing.
    Rare, hard classes dominate gradient updates naturally.
    No manual class weights or sampler needed.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# --------------- Transforms ---------------
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-45, 45), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --------------- Stratified split ---------------
index_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH)
all_labels    = [label for _, label in index_dataset.samples]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(range(len(all_labels)), all_labels))

train_full = ISICClassificationDataset(IMAGE_DIR, CSV_PATH, transform=train_transform)
val_full   = ISICClassificationDataset(IMAGE_DIR, CSV_PATH, transform=val_transform)

train_dataset = Subset(train_full, train_idx)
val_dataset   = Subset(val_full,   val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

val_labels  = [all_labels[i] for i in val_idx]
train_labels = [all_labels[i] for i in train_idx]
val_counts  = {CLASS_NAMES[c]: val_labels.count(c) for c in range(7)}
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"Val class distribution: {val_counts}")

# --------------- Model ---------------
model     = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3").to(device)
criterion = FocalLoss(gamma=FOCAL_GAMMA)
scaler    = GradScaler("cuda")

# --------------- Phase 1: head-only warmup ---------------
print(f"\n--- Phase 1: head warmup ({PHASE1_EPOCHS} epochs, backbone frozen) ---")
model.freeze_backbone()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)

best_val_f1 = 0.0

def run_epoch(phase):
    loader     = train_loader if phase == "train" else val_loader
    is_train   = phase == "train"
    model.train() if is_train else model.eval()

    all_preds, all_targets = [], []
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc=f"  [{phase}]", leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                with autocast("cuda"):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast("cuda"):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            total_loss += loss.item()

    acc    = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return acc, macro_f1, total_loss / len(loader)

for epoch in range(PHASE1_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")
    scheduler.step()
    print(f"Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")

# --------------- Phase 2: full fine-tune ---------------
print(f"\n--- Phase 2: full fine-tune ({PHASE2_EPOCHS} epochs, all layers) ---")
model.unfreeze_all()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

for epoch in range(PHASE2_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")
    scheduler.step()

    ep_global = PHASE1_EPOCHS + epoch + 1
    print(f"Epoch {ep_global:2d}/{PHASE1_EPOCHS+PHASE2_EPOCHS} | loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")

# --------------- Final report ---------------
print(f"\nBest Val Macro-F1: {best_val_f1:.4f}")
print("\nPer-class breakdown (last epoch val):")

model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
all_preds, all_targets = [], []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        with autocast("cuda"):
            outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_targets.extend(labels.numpy())

print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))
print(f"Model saved to: {SAVE_PATH}")
