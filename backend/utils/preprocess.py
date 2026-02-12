import cv2
import torch
import numpy as np

IMG_SIZE = 224
MIN_DIMENSION = 50
BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 30
BRIGHT_THRESHOLD = 240


def assess_image_quality(image_bgr):
    """
    Assess image quality before prediction.
    Returns: dict with 'acceptable' (bool) and 'issues' (list of strings).
    """
    issues = []

    h, w = image_bgr.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        issues.append(f"Image too small ({w}x{h}px, minimum {MIN_DIMENSION}px)")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance as blur metric
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        issues.append(f"Image appears blurry (sharpness: {laplacian_var:.0f})")

    mean_brightness = gray.mean()
    if mean_brightness < DARK_THRESHOLD:
        issues.append(f"Image is too dark (brightness: {mean_brightness:.0f}/255)")
    elif mean_brightness > BRIGHT_THRESHOLD:
        issues.append(f"Image is overexposed (brightness: {mean_brightness:.0f}/255)")

    return {
        "acceptable": len(issues) == 0,
        "issues": issues,
    }


def preprocess_image(image_bytes, device):
    """
    Decode, validate, and preprocess image bytes into a model-ready tensor.
    Returns: (tensor, quality_info) tuple.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return None, {"acceptable": False, "issues": ["Failed to decode image file"]}

    quality = assess_image_quality(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_rgb = image_rgb / 255.0

    tensor = torch.tensor(image_rgb, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device), quality
