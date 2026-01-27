import cv2
import numpy as np

def estimate_skin_tone(image):
    """
    Estimate skin tone using average brightness (simple proxy).
    Returns: 'light', 'medium', or 'dark'
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)

    if mean_intensity > 160:
        return "light"
    elif mean_intensity > 100:
        return "medium"
    else:
        return "dark"
