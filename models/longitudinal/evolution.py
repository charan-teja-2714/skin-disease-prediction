import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
from models.explainability.abcde import color_variation, diameter_mm

# ---------- Utility Functions ----------

def lesion_area(mask):
    """Compute lesion area in pixels"""
    return np.sum(mask > 0)

def percentage_change(old, new):
    if old == 0:
        return 0
    return ((new - old) / old) * 100

# ---------- Evolution Analysis ----------

def analyze_evolution(image_t1, mask_t1, image_t2, mask_t2):
    # Area
    area_t1 = lesion_area(mask_t1)
    area_t2 = lesion_area(mask_t2)
    area_change = percentage_change(area_t1, area_t2)

    # Diameter
    diameter_t1 = diameter_mm(mask_t1)
    diameter_t2 = diameter_mm(mask_t2)
    diameter_change = diameter_t2 - diameter_t1

    # Color variation
    color_t1 = color_variation(image_t1, mask_t1)
    color_t2 = color_variation(image_t2, mask_t2)
    color_change = color_t2 - color_t1

    # Evolution Alert Logic (Clinical Heuristics)
    # Note: color_variation returns 0-1 range, so threshold is 0.3 (not 2)
    alert = False
    if area_change > 20 or diameter_change > 2 or color_change >= 0.3:
        alert = True

    return {
        "Area Change (%)": round(area_change, 2),
        "Diameter Change (mm)": round(diameter_change, 2),
        "Color Change": round(color_change, 3),
        "Evolution Alert": "YES" if alert else "NO"
    }

# ---------- Demo / Test ----------

if __name__ == "__main__":
    # Replace with real image & mask paths
    img1 = cv2.imread("../../data/raw/images/ISIC_0000000.jpg")
    img2 = cv2.imread("../../data/raw/images/ISIC_0000001.jpg")

    mask1 = cv2.imread("../../data/raw/masks/ISIC_0000000_segmentation.png", cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread("../../data/raw/masks/ISIC_0000001_segmentation.png", cv2.IMREAD_GRAYSCALE)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    results = analyze_evolution(img1, mask1, img2, mask2)

    print("\n📈 Longitudinal Evolution Analysis")
    print("---------------------------------")
    for k, v in results.items():
        print(f"{k}: {v}")
