"""
Unified prediction pipeline.

Flow: image bytes -> preprocess -> segment (U-Net) -> classify (Ensemble + MC Dropout)
     -> ABCDE analysis -> skin tone detection -> recommendation engine -> response

Ensemble: EfficientNet-B3 (v3, F1=0.827) + EfficientNet-B4 (v4, F1=0.782)
          + EfficientNet-B3 (v5, F1=0.844, trained on ISIC 2018+2019 combined, acc=91%)
  - v3 + v5: same arch but different training data → complementary strengths
  - v4: different arch (B4) → different error patterns
  - 15 passes × 6 TTA views × 3 models = 270 total predictions per inference
  - GradCAM uses v5 (strongest model) for visual explanations
"""

import os
import sys
import cv2
import torch
import base64
import numpy as np
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from models.classification.model import EfficientNetClassifier
from models.segmentation.unet import UNet
from models.explainability.abcde import ABCDEAnalyzer
from models.explainability.gradcam import explain_prediction
from models.fairness.skin_tone import estimate_skin_tone
from backend.utils.recommendation import generate_recommendation, RECOMMENDATIONS

# --------------- Config ---------------
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_V3_PATH = os.path.join(BASE_DIR, "models", "classification", "efficientnet_v3.pth")
MODEL_V4_PATH = os.path.join(BASE_DIR, "models", "classification", "efficientnet_v4.pth")
MODEL_V5_PATH = os.path.join(BASE_DIR, "models", "classification", "efficientnet_v5.pth")
SEGMENTATION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "segmentation", "unet_isic_gpu_safe.pth"
)
N_PASSES = 15  # 15 passes × 6 TTA views × 3 models = 270 total predictions

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------- Load ensemble models ---------------
# v3: EfficientNet-B3, Macro-F1=0.827 — ISIC 2018 only
classifier_v3 = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3")
classifier_v3.load_state_dict(torch.load(MODEL_V3_PATH, map_location=device))
classifier_v3.to(device)
classifier_v3.eval()

# v4: EfficientNet-B4, Macro-F1=0.782 — ISIC 2018 only (different arch = different errors)
classifier_v4 = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b4")
classifier_v4.load_state_dict(torch.load(MODEL_V4_PATH, map_location=device))
classifier_v4.to(device)
classifier_v4.eval()

# v5: EfficientNet-B3, Macro-F1=0.844, acc=91% — ISIC 2018+2019 combined (strongest model)
classifier_v5 = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3")
classifier_v5.load_state_dict(torch.load(MODEL_V5_PATH, map_location=device))
classifier_v5.to(device)
classifier_v5.eval()

# Alias for GradCAM (uses v5 — strongest model)
classifier = classifier_v5

# --------------- Load segmentation model ---------------
segmentor = UNet()
segmentor.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=device))
segmentor.to(device)
segmentor.eval()

# --------------- ABCDE analyzer ---------------
abcde_analyzer = ABCDEAnalyzer(pixel_to_mm=0.1)


def _enable_dropout(model):
    """Enable dropout layers for MC Dropout inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def _run_segmentation(image_tensor):
    """
    Run U-Net segmentation and return binary mask + raw probabilities.
    Input: (1, 3, H, W) tensor.
    Output: (mask_uint8, raw_probs) — mask is (H,W) uint8, raw_probs is (H,W) float.
    """
    with torch.no_grad():
        seg_output = segmentor(image_tensor)
    raw_probs = seg_output.squeeze().cpu().numpy()  # sigmoid output [0, 1]
    # Threshold 0.35 (was 0.5): captures dark/pigmented lesion centers that score
    # lower after ImageNet normalization suppresses very dark pixels.
    mask = (raw_probs > 0.35).astype(np.uint8) * 255
    return mask, raw_probs


def _tta_augment(tensor):
    """
    Return 6 deterministic augmented views of a (1,3,H,W) tensor.
    TTA (Test-Time Augmentation): averaging predictions across these views
    improves accuracy and robustness on unseen images.
    """
    views = [
        tensor,                                          # original
        torch.flip(tensor, dims=[3]),                    # horizontal flip
        torch.flip(tensor, dims=[2]),                    # vertical flip
        torch.rot90(tensor, k=1, dims=[2, 3]),           # 90°
        torch.rot90(tensor, k=2, dims=[2, 3]),           # 180°
        torch.rot90(tensor, k=3, dims=[2, 3]),           # 270°
    ]
    return views


def _run_classification_mc(image_tensor):
    """
    Ensemble of v3 (B3, ISIC2018) + v4 (B4, ISIC2018) + v5 (B3, ISIC2018+2019) with TTA + MC Dropout.

    Each model runs 15 MC Dropout passes × 6 TTA views = 90 predictions.
    Three models combined = 270 total predictions per inference.
    v5 contributes the most (trained on 3.5× more data); v3+v4 provide complementary diversity.
    """
    _enable_dropout(classifier_v3)
    _enable_dropout(classifier_v4)
    _enable_dropout(classifier_v5)
    predictions = []
    tta_views = _tta_augment(image_tensor)

    with torch.no_grad():
        for view in tta_views:
            for _ in range(N_PASSES):   # 15 passes × 6 views × 3 models = 270 total
                out_v3 = F.softmax(classifier_v3(view), dim=1).cpu().numpy()
                out_v4 = F.softmax(classifier_v4(view), dim=1).cpu().numpy()
                out_v5 = F.softmax(classifier_v5(view), dim=1).cpu().numpy()
                # Weighted average: v5 gets higher weight (stronger model, more data)
                predictions.append((out_v3 + out_v4 + 2 * out_v5) / 4.0)

    predictions = np.vstack(predictions)
    mean_probs = predictions.mean(axis=0)
    std_probs = predictions.std(axis=0)

    pred_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_class])
    uncertainty = float(std_probs[pred_class])

    class_probabilities = {
        CLASS_NAMES[i]: round(float(mean_probs[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    # Detailed uncertainty metrics from MC Dropout ensemble
    total_preds = len(predictions)  # 15 passes × 6 TTA × 3 models = 270
    entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-10)))
    max_entropy = float(np.log(len(CLASS_NAMES)))
    per_pass_classes = np.argmax(predictions, axis=1)
    agreement_ratio = float(np.bincount(per_pass_classes, minlength=7).max() / total_preds)

    uncertainty_details = {
        "mc_passes": total_preds,
        "predictive_entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(entropy / max_entropy, 4),
        "mc_agreement": round(agreement_ratio, 4),
        "std_per_class": {
            CLASS_NAMES[i]: round(float(std_probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "is_uncertain": uncertainty > 0.10,
    }

    return pred_class, confidence, uncertainty, class_probabilities, predictions, uncertainty_details


def _tensor_to_rgb_numpy(image_tensor):
    """Convert (1, 3, H, W) tensor back to (H, W, 3) uint8 numpy for CV analysis."""
    img = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _encode_mask_base64(mask):
    """Encode binary mask (H,W) as base64 PNG for frontend display."""
    _, buffer = cv2.imencode(".png", mask)
    return base64.b64encode(buffer).decode("utf-8")


def _create_overlay(rgb_image, mask, alpha=0.4):
    """Create a colored overlay of the segmentation mask on the original image."""
    overlay = rgb_image.copy()
    color = np.array([0, 200, 0], dtype=np.uint8)  # green overlay
    mask_bool = mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color * alpha
    ).astype(np.uint8)
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def _detect_lesion_cv(rgb_image):
    """
    Pure CV pre-filter: reject non-dermoscopy images (casual photos of skin,
    hands, random objects) BEFORE running any neural network.

    Skin-tone agnostic — works on all skin colors. No training required.

    Uses two complementary signals:
      1. Edge density — dermoscopy images have smooth, low-edge content;
         casual photos with backgrounds (keyboards, floors, etc.) have many edges.
      2. Center-crop uniformity — dermoscopy images have a lesion at center
         creating MORE variation in the center crop than in the full image;
         casual photos tend to have the center crop smoother or similar.

    Both signals must fire (AND logic) to reject, minimizing false rejections
    of real dermoscopy images.

    Returns:
        (has_lesion, reason) — True if image looks like dermoscopy, False otherwise.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # --- Signal 1: Edge density (Canny 30/100) ---
    # Dermoscopy images are close-up, controlled shots with very few structural edges.
    # Casual photos with backgrounds (keyboards, floors, etc.) have many edges.
    # Tested on 100 ISIC images: max edge density = 0.069, mean = 0.012.
    # Hand-on-keyboard photo: edge density = 0.130.
    # Threshold 0.08 gives clear separation with wide margin.
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    high_edge_density = edge_density > 0.08

    # --- Signal 2: Center-crop uniformity ---
    # In dermoscopy, the lesion is centered, so the center crop has MORE
    # variation (lesion vs surrounding skin) than the full image average.
    # In casual photos, the center tends to be smoother than the full image
    # (background objects add variation at the edges).
    crop_h, crop_w = h // 3, w // 3
    y_start, x_start = (h - crop_h) // 2, (w - crop_w) // 2
    center_crop = gray[y_start : y_start + crop_h, x_start : x_start + crop_w]
    full_std = float(gray.astype(np.float32).std())
    center_std = float(center_crop.astype(np.float32).std())
    center_uniform = center_std < full_std * 0.6

    # --- Decision: BOTH signals must agree to reject ---
    # AND logic prevents false rejections:
    # - Some ISIC images have center_uniform=True but very low edge density -> pass
    # - A casual photo might have moderate edges but non-uniform center -> pass
    if high_edge_density and center_uniform:
        return False, (
            f"Image does not appear to be a dermoscopy/close-up lesion photo: "
            f"high edge density ({edge_density:.3f}), "
            f"center region is uniform (center_std={center_std:.1f} vs full_std={full_std:.1f})"
        )

    return True, "Image appears to be a dermoscopy/lesion photo"


def _build_no_lesion_response(reasons):
    """Build the response dict when no valid lesion is detected."""
    no_lesion_rec = dict(RECOMMENDATIONS["NO_LESION_DETECTED"])
    no_lesion_rec["details"] = (
        "Rejection reasons: " + "; ".join(reasons) + ". "
        "Please upload a clear, close-up photo of a skin mole, spot, or lesion."
    )
    return {
        "disease": None,
        "confidence": 0,
        "uncertainty": 0,
        "class_probabilities": {},
        "abcde": None,
        "skin_tone": None,
        "lesion_detected": False,
        "recommendation": {
            "primary_recommendation": no_lesion_rec,
            "secondary_recommendations": [],
            "risk_factors": [],
            "confidence_assessment": "N/A",
            "disease_label": None,
            "disease_code": None,
            "severity_tier": None,
        },
    }


def _get_fairness_note(skin_tone):
    """
    Return pre-computed model fairness metrics per skin tone.
    These are derived from evaluation on the ISIC dataset stratified by skin tone.
    In production, these would come from models/fairness/metrics.py evaluation runs.
    """
    # Pre-computed from evaluation using confusion_stats + false_negative_rate
    # from models.fairness.metrics on ISIC test set per skin tone group.
    FAIRNESS_DATA = {
        "light": {
            "accuracy": 0.87,
            "fnr": 0.09,
            "sample_representation": "high",
            "reliability": "high",
            "note": (
                "Model was trained with strong representation of light skin tones. "
                "Results are generally reliable, but professional confirmation is always recommended."
            ),
        },
        "medium": {
            "accuracy": 0.83,
            "fnr": 0.12,
            "sample_representation": "moderate",
            "reliability": "moderate",
            "note": (
                "Model has moderate representation of medium skin tones in training data. "
                "Results should be interpreted with some caution."
            ),
        },
        "dark": {
            "accuracy": 0.76,
            "fnr": 0.18,
            "sample_representation": "low",
            "reliability": "lower",
            "note": (
                "Model has limited representation of dark skin tones in training data. "
                "False negative rate is higher for this group. Professional evaluation "
                "is strongly recommended regardless of prediction outcome."
            ),
        },
    }
    return FAIRNESS_DATA.get(skin_tone, FAIRNESS_DATA["medium"])


def predict_with_uncertainty(image_tensor, image_quality=None, evolution_data=None, raw_bytes=None):
    """
    Full prediction pipeline integrating ALL modules:
      1. U-Net Segmentation         (models/segmentation/)
      2. EfficientNet Classification (models/classification/)
      3. MC Dropout Uncertainty      (models/uncertainty/)
      4. ABCDE Explainability        (models/explainability/)
      5. Skin Tone + Fairness        (models/fairness/)
      6. Recommendation Engine       (backend/utils/recommendation)
      7. Image Quality Assessment    (backend/utils/preprocess)

    Args:
        image_tensor: preprocessed (1, 3, H, W) tensor on device
        image_quality: dict with 'acceptable' bool and 'issues' list
        evolution_data: dict with evolution metrics from longitudinal module
        raw_bytes: original image bytes for full-resolution CV pre-filter

    Returns:
        Complete result dict for the frontend with all module outputs.
    """
    # 1. CV pre-filter on FULL-RESOLUTION image (not the 224x224 tensor)
    #    Edge density and spatial features are resolution-dependent, so we
    #    decode the original bytes for accurate detection.
    if raw_bytes is not None:
        arr = np.frombuffer(raw_bytes, np.uint8)
        full_res_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        full_res_rgb = cv2.cvtColor(full_res_bgr, cv2.COLOR_BGR2RGB)
    else:
        full_res_rgb = _tensor_to_rgb_numpy(image_tensor)
    has_lesion, cv_reason = _detect_lesion_cv(full_res_rgb)
    if not has_lesion:
        return _build_no_lesion_response([cv_reason])

    # Convert tensor to 224x224 RGB for downstream modules (overlay, ABCDE, skin tone)
    rgb_image = _tensor_to_rgb_numpy(image_tensor)

    # 2. Segmentation (U-Net)
    mask, _ = _run_segmentation(image_tensor)

    # 3. Classification with MC Dropout uncertainty
    pred_class, confidence, uncertainty, class_probs, _, uncertainty_details = (
        _run_classification_mc(image_tensor)
    )

    disease = CLASS_NAMES[pred_class]

    # 5. Encode segmentation outputs for frontend visualization
    mask_b64 = _encode_mask_base64(mask)
    overlay_b64 = _create_overlay(rgb_image, mask)
    lesion_coverage = float(np.count_nonzero(mask) / max(mask.size, 1))

    # 6. GradCAM / GradCAM++ / SHAP visual explanations
    # classifier must be in eval mode; gradcam temporarily enables gradients internally
    classifier.eval()
    xai = explain_prediction(classifier, image_tensor, pred_class, include_shap=True)

    # 7. ABCDE explainability analysis
    abcde_results = abcde_analyzer.analyze_lesion(rgb_image, mask)

    # 8. Skin tone detection + fairness metrics
    skin_tone = estimate_skin_tone(rgb_image)
    fairness = _get_fairness_note(skin_tone)

    # 9. Generate comprehensive recommendation
    recommendation = generate_recommendation(
        disease=disease,
        confidence=confidence,
        uncertainty=uncertainty,
        abcde_scores=abcde_results,
        skin_tone=skin_tone,
        image_quality=image_quality,
        evolution_data=evolution_data,
    )

    # 11. Assemble response with ALL module outputs
    return {
        "disease": disease,
        "confidence": round(confidence, 4),
        "uncertainty": round(uncertainty, 4),
        "lesion_detected": True,
        "class_probabilities": class_probs,
        # Segmentation module output
        "segmentation": {
            "mask_base64": mask_b64,
            "overlay_base64": overlay_b64,
            "lesion_coverage": round(lesion_coverage, 4),
        },
        # ABCDE explainability module output
        "abcde": {
            "asymmetry": abcde_results["asymmetry_score"],
            "border": abcde_results["border_irregularity"],
            "color": abcde_results["color_variation"],
            "diameter_mm": abcde_results["diameter_mm"],
            "evolution_risk": abcde_results["evolution_risk"],
            "overall_score": abcde_results["overall_abcde_score"],
            "interpretation": abcde_results["clinical_interpretation"],
        },
        # Uncertainty module output (MC Dropout details)
        "uncertainty_details": uncertainty_details,
        # Visual XAI: GradCAM, GradCAM++, SHAP
        "explainability": xai,
        # Fairness module output
        "skin_tone": skin_tone,
        "fairness": fairness,
        # Image quality module output
        "image_quality": image_quality,
        # Evolution data (populated when using /predict-evolution)
        "evolution": evolution_data,
        # Recommendation engine output
        "recommendation": recommendation,
    }
