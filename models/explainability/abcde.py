import cv2
import numpy as np

# ---------- A: Asymmetry ----------
def asymmetry_score(mask):
    h, w = mask.shape
    left = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])

    diff = np.abs(left[:, :right.shape[1]] - right)
    score = np.sum(diff) / np.sum(mask)
    return round(score, 3)

# ---------- B: Border Irregularity ----------
def border_irregularity(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)

    if area == 0:
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    irregularity = 1 - circularity
    return round(irregularity, 3)

# ---------- C: Color Variation ----------
def color_variation(image, mask, k=3):
    lesion_pixels = image[mask > 0]
    if len(lesion_pixels) < 10:
        return 1

    lesion_pixels = lesion_pixels.reshape(-1, 3)
    lesion_pixels = np.float32(lesion_pixels)

    _, labels, _ = cv2.kmeans(
        lesion_pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    unique_colors = len(np.unique(labels))
    return unique_colors

# ---------- D: Diameter ----------
def diameter_mm(mask, pixel_to_mm=0.1):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter_px = radius * 2
    diameter_mm = diameter_px * pixel_to_mm

    return round(diameter_mm, 2)
