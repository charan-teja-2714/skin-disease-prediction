import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import ISICSegmentationDataset
from unet import UNet

IMAGE_DIR = "../../data/raw/images"
MASK_DIR = "../../data/raw/masks"
MODEL_PATH = "unet_isic_gpu_safe.pth"
BATCH_SIZE = 8
IMG_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# Load dataset
dataset = ISICSegmentationDataset(IMAGE_DIR, MASK_DIR, img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"📊 Evaluating on {len(dataset)} images...")

def dice_coefficient(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

# Evaluate
dice_scores = []
iou_scores = []
pixel_accs = []

with torch.no_grad():
    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        
        for i in range(images.size(0)):
            dice = dice_coefficient(preds[i], masks[i])
            iou = iou_score(preds[i], masks[i])
            pixel_acc = pixel_accuracy(preds[i], masks[i])
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            pixel_accs.append(pixel_acc.item())

# Calculate metrics
avg_dice = np.mean(dice_scores)
avg_iou = np.mean(iou_scores)
avg_pixel_acc = np.mean(pixel_accs)

std_dice = np.std(dice_scores)
std_iou = np.std(iou_scores)
std_pixel_acc = np.std(pixel_accs)

print("\n" + "="*50)
print("📊 SEGMENTATION EVALUATION RESULTS")
print("="*50)
print(f"Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}")
print(f"IoU Score:        {avg_iou:.4f} ± {std_iou:.4f}")
print(f"Pixel Accuracy:   {avg_pixel_acc:.4f} ± {std_pixel_acc:.4f}")
print("="*50)

# Save results
with open("segmentation_results.txt", "w") as f:
    f.write("SEGMENTATION MODEL EVALUATION\n")
    f.write("="*50 + "\n\n")
    f.write(f"Dataset: {len(dataset)} images\n")
    f.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"Device: {device}\n\n")
    f.write("METRICS:\n")
    f.write("-"*30 + "\n")
    f.write(f"Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}\n")
    f.write(f"IoU Score:        {avg_iou:.4f} ± {std_iou:.4f}\n")
    f.write(f"Pixel Accuracy:   {avg_pixel_acc:.4f} ± {std_pixel_acc:.4f}\n")

print("\n✅ Results saved to segmentation_results.txt")
