"""
EfficientNet-B4 training script — v4.

Architecture : EfficientNet-B4 (19M params — more capacity for subtle skin features)
Loss         : Focal Loss (gamma=2) + Label Smoothing (0.1)
               Label smoothing prevents overconfident predictions, improves calibration.
               Focal Loss handles the NV class dominance.
Augmentation : Heavy spatial + color transforms (albumentations). No MixUp — ISIC 2018 is too
               small and imbalanced; MixUp blends MEL with NV and collapses MEL recall.
Loss         : Focal Loss only (gamma=2) — no class weights. Focal handles imbalance by
               down-weighting easy NV examples. Adding class weights on top double-corrects
               and collapses overall accuracy.
Training     : Two-phase fine-tuning
               Phase 1 (epochs 1-10)  : backbone frozen, train head only at 1e-3
               Phase 2 (epochs 11-50) : full model, AdamW + cosine LR 1e-4 → 1e-6
Precision    : Mixed precision (AMP) — 2x faster on GPU
Output       : efficientnet_v4.pth (replaces v3 as production model)
"""

import os
import sys
import time
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
SAVE_PATH  = os.path.join(os.path.dirname(__file__), "efficientnet_v4.pth")
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

PHASE1_EPOCHS = 10   # head-only warmup
PHASE2_EPOCHS = 40   # full fine-tune  (total = 50)
BATCH_SIZE    = 32   # 32 for stable gradients (B4 fits in VRAM at 224x224)
FOCAL_GAMMA   = 2.0
LABEL_SMOOTH  = 0.05  # Mild smoothing only — prevent overconfidence without hurting accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

training_start = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# --------------- Focal Loss + Label Smoothing ---------------
class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing.

    Label smoothing replaces hard 1-hot targets with soft targets:
        y_smooth = (1 - eps) * y_hard + eps / num_classes
    This prevents the model from being overconfident, improving calibration
    and generalization — especially important for clinical decision support.

    Combined with Focal Loss for handling class imbalance.
    """
    def __init__(self, gamma=2.0, smoothing=0.05, num_classes=7):
        super().__init__()
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, label_smoothing=self.smoothing, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


# --------------- Transforms ---------------
train_transform = A.Compose([
    A.Resize(224, 224),   # Match inference preprocess.py (consistent pipeline)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-45, 45), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.ElasticTransform(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
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

train_labels = [all_labels[i] for i in train_idx]
val_labels   = [all_labels[i] for i in val_idx]
val_counts   = {CLASS_NAMES[c]: val_labels.count(c) for c in range(7)}
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"Val class distribution: {val_counts}")

# --------------- Model ---------------
model     = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b4").to(device)
criterion = FocalLossWithSmoothing(gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH)
scaler    = GradScaler("cuda")

best_val_f1 = 0.0


# --------------- Training loop ---------------
def run_epoch(phase):
    loader   = train_loader if phase == "train" else val_loader
    is_train = phase == "train"
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    acc      = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return acc, macro_f1, total_loss / len(loader)


# --------------- Phase 1: head-only warmup ---------------
print(f"\n--- Phase 1: head warmup ({PHASE1_EPOCHS} epochs, backbone frozen) ---")
model.freeze_backbone()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)

for epoch in range(PHASE1_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")
    scheduler.step()
    print(
        f"Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | "
        f"loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | "
        f"val acc={va_acc:.4f} f1={va_f1:.4f}"
    )
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")


# --------------- Phase 2: full fine-tune ---------------
print(f"\n--- Phase 2: full fine-tune ({PHASE2_EPOCHS} epochs, all layers) ---")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))  # start from best Phase 1
model.unfreeze_all()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# AdamW with layer-wise learning rates:
# backbone at lower LR (avoid destroying pretrained features)
# head at higher LR (faster adaptation)
backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params     = [p for n, p in model.named_parameters() if "classifier" in n]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params,     "lr": 5e-4},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

for epoch in range(PHASE2_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")
    scheduler.step()

    ep_global = PHASE1_EPOCHS + epoch + 1
    print(
        f"Epoch {ep_global:2d}/{PHASE1_EPOCHS+PHASE2_EPOCHS} | "
        f"loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | "
        f"val acc={va_acc:.4f} f1={va_f1:.4f}"
    )
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")


# --------------- Final report ---------------
print(f"\nBest Val Macro-F1: {best_val_f1:.4f}")
print("\nPer-class breakdown on validation set:")

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

elapsed = time.time() - training_start
hours, rem = divmod(int(elapsed), 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTraining started at : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start))}")
print(f"Training ended at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time : {hours:02d}h {minutes:02d}m {seconds:02d}s")
