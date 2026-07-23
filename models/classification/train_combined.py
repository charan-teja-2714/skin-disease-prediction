"""
Combined ISIC 2018 + ISIC 2019 training script — v5.

Data         : ISIC 2018 (10,015 images) + ISIC 2019 (25,331 images) = ~35,000 total
               ISIC 2019 AK + SCC → mapped to AKIEC (same actinic spectrum)
               Overlap handling: ISIC 2019 images already in ISIC 2018 val set are excluded

Architecture : EfficientNet-B3 (proven best on this task — 88.2% on ISIC 2018 alone)
               Warm-started from efficientnet_v3.pth (already learned ISIC 2018 features)
Loss         : Focal Loss (gamma=2) — same proven formula as v3, no class weights
Training     : Two-phase fine-tuning with warm-start (converges ~2x faster than cold start)
               Phase 1 (epochs 1-3)   : backbone frozen, OneCycleLR head warmup lr=1e-3
               Phase 2 (epochs 4-20)  : full model, OneCycleLR 1e-4 → 0
Output       : efficientnet_v5.pth (combined dataset model, expected ~89-91%)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from models.classification.model import EfficientNetClassifier

# --------------- Config ---------------
# ISIC 2018 (HAM10000)
DIR_2018   = os.path.join(PROJECT_ROOT, "data/raw/images")
CSV_2018   = os.path.join(PROJECT_ROOT, "data/raw/metadata.csv")

# ISIC 2019
DIR_2019   = os.path.join(PROJECT_ROOT, "data/isic2019/images/ISIC_2019_Training_Input")
CSV_2019   = os.path.join(PROJECT_ROOT, "data/isic2019/ISIC_2019_Training_GroundTruth.csv")

WARM_START_PATH = os.path.join(os.path.dirname(__file__), "efficientnet_v3.pth")
SAVE_PATH       = os.path.join(os.path.dirname(__file__), "efficientnet_v5.pth")
CLASS_NAMES     = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

PHASE1_EPOCHS = 3    # head warmup — short because warm-started from v3
PHASE2_EPOCHS = 17   # full fine-tune (total = 20)
BATCH_SIZE    = 48   # B3 uses less VRAM than B4 → bigger batch → fewer steps/epoch
FOCAL_GAMMA   = 2.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

training_start = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# --------------- Focal Loss ---------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# --------------- Dataset ---------------
class ISICDataset(Dataset):
    """
    Unified loader for ISIC 2018 and 2019 with class mapping.

    ISIC 2019 column mapping to ISIC 2018 CLASS_NAMES:
      MEL, NV, BCC, BKL, DF, VASC → direct
      AK  → AKIEC (Actinic Keratosis, same clinical entity)
      SCC → AKIEC (Squamous Cell Carcinoma, part of AKIEC spectrum)
      UNK → excluded
    """
    COL_MAP_2019 = {
        "MEL": "MEL", "NV": "NV", "BCC": "BCC",
        "AK": "AKIEC", "SCC": "AKIEC",
        "BKL": "BKL", "DF": "DF", "VASC": "VASC",
    }

    def __init__(self, image_dir, csv_path, transform=None, is_2019=False, exclude_ids=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples   = []

        df = pd.read_csv(csv_path)
        exclude_ids = set(exclude_ids or [])

        for _, row in df.iterrows():
            image_id = row["image"]
            if image_id in exclude_ids:
                continue

            label = None
            if is_2019:
                for col, cls_name in self.COL_MAP_2019.items():
                    if col in row and row[col] == 1:
                        label = CLASS_NAMES.index(cls_name)
                        break
            else:
                for idx, cls in enumerate(CLASS_NAMES):
                    if cls in row and row[cls] == 1:
                        label = idx
                        break

            if label is None:
                continue

            img_path = os.path.join(image_dir, image_id + ".jpg")
            if os.path.exists(img_path):
                self.samples.append((img_path, label))

        print(f"  Loaded {len(self.samples)} samples from {os.path.basename(csv_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)


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


# --------------- Build datasets ---------------
print("\nBuilding datasets...")

# ISIC 2018: stratified split (same seed as v3/v4 — keeps val set consistent)
ds_2018_index = ISICDataset(DIR_2018, CSV_2018)
all_labels_2018 = [label for _, label in ds_2018_index.samples]
all_ids_2018    = [os.path.splitext(os.path.basename(p))[0] for p, _ in ds_2018_index.samples]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx_2018, val_idx_2018 = next(splitter.split(range(len(all_labels_2018)), all_labels_2018))

# Val set image IDs — used to exclude overlapping 2019 images from training
val_ids_2018 = {all_ids_2018[i] for i in val_idx_2018}

ds_2018_train = ISICDataset(DIR_2018, CSV_2018, transform=train_transform)
ds_2018_val   = ISICDataset(DIR_2018, CSV_2018, transform=val_transform)

train_2018 = Subset(ds_2018_train, train_idx_2018)
val_2018   = Subset(ds_2018_val,   val_idx_2018)

# ISIC 2019: all images except those in 2018 val set (avoid leakage)
print("Loading ISIC 2019 (excluding 2018 val images)...")
ds_2019_train = ISICDataset(DIR_2019, CSV_2019, transform=train_transform,
                             is_2019=True, exclude_ids=val_ids_2018)

# Combined training set
train_combined = ConcatDataset([train_2018, ds_2019_train])

train_loader = DataLoader(train_combined, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True, persistent_workers=False)
val_loader   = DataLoader(val_2018,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

print(f"\nTrain: {len(train_combined):,}  ({len(train_2018):,} from 2018 + {len(ds_2019_train):,} from 2019)")
print(f"Val  : {len(val_2018):,} (ISIC 2018 val set only — consistent benchmark)")

val_labels   = [all_labels_2018[i] for i in val_idx_2018]
val_counts   = {CLASS_NAMES[c]: val_labels.count(c) for c in range(7)}
print(f"Val class distribution: {val_counts}")


# --------------- Model (warm-start from v3) ---------------
model = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3").to(device)
if os.path.exists(WARM_START_PATH):
    model.load_state_dict(torch.load(WARM_START_PATH, map_location=device))
    print(f"Warm-started from {WARM_START_PATH}")
else:
    print(f"WARNING: {WARM_START_PATH} not found — training from ImageNet init")

criterion   = FocalLoss(gamma=FOCAL_GAMMA)
scaler      = GradScaler("cuda")
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
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # OneCycleLR steps per batch
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


# --------------- Phase 1: head warmup ---------------
print(f"\n--- Phase 1: head warmup ({PHASE1_EPOCHS} epochs, backbone frozen) ---")
model.freeze_backbone()
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4
)
# OneCycleLR: ramps up then decays within each phase — faster convergence than cosine
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader), epochs=PHASE1_EPOCHS,
    pct_start=0.3, div_factor=10, final_div_factor=100
)

for epoch in range(PHASE1_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")
    print(f"Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")


# --------------- Phase 2: full fine-tune ---------------
print(f"\n--- Phase 2: full fine-tune ({PHASE2_EPOCHS} epochs, all layers) ---")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.unfreeze_all()
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Layer-wise LR: backbone at lower rate to preserve pretrained features
backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params     = [p for n, p in model.named_parameters() if "classifier" in n]

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params,     "lr": 5e-4},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[1e-4, 5e-4],
    steps_per_epoch=len(train_loader), epochs=PHASE2_EPOCHS,
    pct_start=0.1, div_factor=10, final_div_factor=1000
)

for epoch in range(PHASE2_EPOCHS):
    tr_acc, tr_f1, tr_loss = run_epoch("train")
    va_acc, va_f1, va_loss = run_epoch("val")

    ep_global = PHASE1_EPOCHS + epoch + 1
    print(f"Epoch {ep_global:2d}/{PHASE1_EPOCHS+PHASE2_EPOCHS} | loss={tr_loss:.4f} | train acc={tr_acc:.4f} f1={tr_f1:.4f} | val acc={va_acc:.4f} f1={va_f1:.4f}")
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   Saved (Macro-F1: {va_f1:.4f})")


# --------------- Final report ---------------
print(f"\nBest Val Macro-F1: {best_val_f1:.4f}")
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
h, rem  = divmod(int(elapsed), 3600)
m, s    = divmod(rem, 60)
print(f"\nTraining started at : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start))}")
print(f"Training ended at   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time : {h:02d}h {m:02d}m {s:02d}s")
