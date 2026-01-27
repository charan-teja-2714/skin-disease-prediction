import cv2
from abcde import (
    asymmetry_score,
    border_irregularity,
    color_variation,
    diameter_mm
)

# 🔹 Use ONE image and ONE mask (from your dataset)
IMAGE_PATH = "../../data/raw/images/ISIC_0000000.jpg"   # change to any valid image
MASK_PATH  = "../../data/raw/masks/ISIC_0000000_segmentation.png"

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

print("ABCDE Explainability Results")
print("----------------------------")
print("Asymmetry Score       :", asymmetry_score(mask))
print("Border Irregularity   :", border_irregularity(mask))
print("Color Variation Zones :", color_variation(image, mask))
print("Diameter (mm)         :", diameter_mm(mask))
