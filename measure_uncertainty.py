"""
measure_uncertainty.py
Runs v5 (EfficientNet-B3, 91% acc) with MC Dropout on the ISIC 2018 validation split
and prints per-class uncertainty metrics for the research paper table.

Usage:
    python measure_uncertainty.py

Output:
    - Console table with exact values
    - uncertainty_results.csv
    - Ready-to-paste LaTeX rows
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from models.classification.model import EfficientNetClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
IMG_SIZE      = 224
N_PASSES      = 15          # 15 MC passes × 6 TTA views = 90 predictions
UNCERTAIN_THR = 0.10        # is_uncertain flag threshold
VAL_FRACTION  = 0.2
SEED          = 42

META_CSV   = os.path.join(PROJECT_ROOT, "data", "raw", "metadata.csv")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "images")
MODEL_V5   = os.path.join(PROJECT_ROOT, "models", "classification", "efficientnet_v5.pth")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load v5 only ────────────────────────────────────────────────────────────────
print("Loading v5 model...")
model = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3")
model.load_state_dict(torch.load(MODEL_V5, map_location=device))
model.to(device).eval()
print("  v5 loaded.")

def enable_dropout(m):
    for mod in m.modules():
        if mod.__class__.__name__.startswith("Dropout"):
            mod.train()

# ── Validation split (same seed/fraction as training) ───────────────────────────
df = pd.read_csv(META_CSV)
df["label"] = df[CLASS_NAMES].values.argmax(axis=1)
df["exists"] = df["image"].apply(
    lambda x: os.path.exists(os.path.join(IMAGES_DIR, x + ".jpg"))
)
df = df[df["exists"]].reset_index(drop=True)

_, val_df = train_test_split(
    df, test_size=VAL_FRACTION, stratify=df["label"], random_state=SEED
)
val_df = val_df.reset_index(drop=True)
print(f"Validation set: {len(val_df)} images\n")

# ── Preprocessing ────────────────────────────────────────────────────────────────
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return t

def tta_batch(t):
    """6 TTA views × N_PASSES repeats → (90, 3, H, W) batch for one forward pass."""
    views = [
        t,
        torch.flip(t, dims=[3]),
        torch.flip(t, dims=[2]),
        torch.rot90(t, k=1, dims=[2, 3]),
        torch.rot90(t, k=2, dims=[2, 3]),
        torch.rot90(t, k=3, dims=[2, 3]),
    ]
    repeated = [v.expand(N_PASSES, -1, -1, -1) for v in views]
    return torch.cat(repeated, dim=0)  # (90, 3, H, W)

# ── Inference ────────────────────────────────────────────────────────────────────
def run_mc(tensor):
    """90 predictions in a single batched forward pass."""
    enable_dropout(model)
    batch = tta_batch(tensor)          # (90, 3, H, W)
    with torch.no_grad():
        preds = F.softmax(model(batch), dim=1).cpu().numpy()  # (90, 7)
    mean_p = preds.mean(axis=0)        # (7,)
    std_p  = preds.std(axis=0)         # (7,)
    return mean_p, std_p, preds

# ── Collect per-sample metrics ───────────────────────────────────────────────────
records = []
N_LOG = 100
t0 = time.time()

for i, row in val_df.iterrows():
    img_path = os.path.join(IMAGES_DIR, row["image"] + ".jpg")
    tensor = preprocess(img_path)
    if tensor is None:
        continue

    mean_p, std_p, preds = run_mc(tensor)
    pred_class = int(np.argmax(mean_p))
    true_class = int(row["label"])

    entropy      = -float(np.sum(mean_p * np.log(mean_p + 1e-10)))
    norm_entropy = entropy / float(np.log(len(CLASS_NAMES)))

    per_pass_cls = np.argmax(preds, axis=1)
    agreement    = float((per_pass_cls == pred_class).mean())

    pred_std     = float(std_p[pred_class])
    is_uncertain = pred_std > UNCERTAIN_THR

    records.append({
        "true":         true_class,
        "pred":         pred_class,
        "norm_entropy": norm_entropy,
        "agreement":    agreement,
        "pred_std":     pred_std,
        "is_uncertain": is_uncertain,
    })

    if len(records) % N_LOG == 0:
        elapsed = time.time() - t0
        remaining = (elapsed / len(records)) * (len(val_df) - len(records))
        print(f"  {len(records)}/{len(val_df)} done  "
              f"({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)")

results_df = pd.DataFrame(records)
results_df.to_csv("uncertainty_results.csv", index=False)
print(f"\nSaved → uncertainty_results.csv  ({len(results_df)} rows)")

# ── Per-class summary ─────────────────────────────────────────────────────────────
print("\n" + "="*78)
print(f"{'Class':<8} {'Norm Entropy':>15}  {'Agreement':>12}  {'Pred Std':>12}  {'Uncertain%':>12}  {'N':>6}")
print("="*78)

class_rows = {}
for ci, cname in enumerate(CLASS_NAMES):
    subset = results_df[results_df["true"] == ci]
    if len(subset) == 0:
        print(f"{cname:<8}  (no samples)")
        continue
    ne_m  = subset["norm_entropy"].mean()
    ne_s  = subset["norm_entropy"].std()
    ag_m  = subset["agreement"].mean()
    ag_s  = subset["agreement"].std()
    st_m  = subset["pred_std"].mean()
    st_s  = subset["pred_std"].std()
    unc_p = subset["is_uncertain"].mean() * 100

    class_rows[cname] = (ne_m, ne_s, ag_m, ag_s, st_m, st_s, unc_p, len(subset))
    print(f"{cname:<8}  {ne_m:.2f} ± {ne_s:.2f}   "
          f"{ag_m:.2f} ± {ag_s:.2f}   "
          f"{st_m:.2f} ± {st_s:.2f}   "
          f"{unc_p:.0f}%   (n={len(subset)})")

print("="*78)

# ── Ready-to-paste LaTeX rows ─────────────────────────────────────────────────────
print("\nLaTeX table rows (paste directly into paper):\n")
for cname, (ne_m, ne_s, ag_m, ag_s, st_m, st_s, unc_p, n) in class_rows.items():
    print(f"{cname:<6} & ${ne_m:.2f} \\pm {ne_s:.2f}$ & "
          f"${ag_m:.2f} \\pm {ag_s:.2f}$ & "
          f"${st_m:.2f} \\pm {st_s:.2f}$ & "
          f"{unc_p:.0f} \\\\")

total_time = (time.time() - t0) / 60
print(f"\nDone in {total_time:.1f} min.")
