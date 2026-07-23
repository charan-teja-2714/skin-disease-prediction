"""
Visual explanation module: GradCAM, GradCAM++, and SHAP.

GradCAM   : Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
GradCAM++ : Improved CAM with better localization (Chattopadhyay et al., 2018)
SHAP      : SHapley Additive exPlanations via GradientExplainer (Lundberg et al., 2017)

All three produce heatmaps showing WHICH pixels drove the model's prediction,
overlaid on the original image and returned as base64 PNGs for the frontend.
"""

import cv2
import base64
import numpy as np
import torch
import torch.nn.functional as F

# pytorch-grad-cam library (pip install grad-cam)
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# --------------- Target layer resolution ---------------

def _get_target_layer(model):
    """
    Return the correct GradCAM target layer for timm EfficientNet-B3.

    WHY conv_head and NOT blocks[-1]:
      - model.model.blocks[-1] is a Sequential (group of MBConv blocks), not a
        single layer. pytorch-grad-cam hooks into the LAST op inside it, which
        produces near-zero gradients → heatmap appears all blue (low activation).
      - model.model.conv_head is the final 1x1 pointwise conv before global
        average pooling. It produces (B, 1536, 7, 7) feature maps with full
        task-specific spatial information — correct for GradCAM.
    """
    return [model.model.conv_head]


# --------------- Image decoding helper ---------------

def _tensor_to_rgb_float(image_tensor):
    """
    Convert (1, 3, H, W) normalized tensor back to (H, W, 3) float32 in [0, 1]
    for overlay rendering. Reverses the ImageNet normalization.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img = img * std + mean          # de-normalize
    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def _encode_overlay(overlay_rgb_float):
    """Encode (H, W, 3) float32 [0,1] RGB image as base64 PNG."""
    img_uint8 = (overlay_rgb_float * 255).clip(0, 255).astype(np.uint8)
    img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _, buf    = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


# --------------- GradCAM ---------------

def run_gradcam(model, image_tensor, pred_class):
    """
    Generate GradCAM heatmap for the predicted class.

    Args:
        model        : EfficientNetClassifier (eval mode)
        image_tensor : (1, 3, 224, 224) preprocessed tensor on device
        pred_class   : int, index of predicted class

    Returns:
        dict with 'heatmap_base64' and 'overlay_base64'
    """
    target_layers = _get_target_layer(model)
    targets       = [ClassifierOutputTarget(pred_class)]
    rgb_float     = _tensor_to_rgb_float(image_tensor)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]  # (H, W)

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Heatmap only (colorized)
    heatmap = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return {
        "heatmap_base64" : _encode_overlay(heatmap_rgb.astype(np.float32) / 255.0),
        "overlay_base64" : _encode_overlay(overlay.astype(np.float32) / 255.0),
    }


# --------------- GradCAM++ ---------------

def run_gradcam_plus(model, image_tensor, pred_class):
    """
    Generate GradCAM++ heatmap. Better than GradCAM for:
    - Multiple instances of the same class in the image
    - Finer localization of discriminative regions

    Same interface as run_gradcam.
    """
    target_layers = _get_target_layer(model)
    targets       = [ClassifierOutputTarget(pred_class)]
    rgb_float     = _tensor_to_rgb_float(image_tensor)

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    heatmap = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return {
        "heatmap_base64" : _encode_overlay(heatmap_rgb.astype(np.float32) / 255.0),
        "overlay_base64" : _encode_overlay(overlay.astype(np.float32) / 255.0),
    }


# --------------- SHAP ---------------

def run_shap(model, image_tensor, pred_class, n_samples=50):
    """
    Generate SHAP explanation using GradientExplainer.

    GradientExplainer is the fastest SHAP variant for deep networks.
    It computes expected gradients — approximating SHAP values using
    a background distribution of n_samples random noise perturbations.

    Args:
        model        : EfficientNetClassifier (eval mode)
        image_tensor : (1, 3, 224, 224) preprocessed tensor on device
        pred_class   : int, index of predicted class
        n_samples    : number of background samples (more = more accurate, slower)

    Returns:
        dict with 'overlay_base64' — red=positive contribution, blue=negative
    """
    import shap

    model.eval()
    device = image_tensor.device

    # Background: random Gaussian noise (represents "absence" of signal)
    background = torch.randn(n_samples, 3, 224, 224, device=device) * 0.1

    explainer   = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)  # list of 7 arrays, each (1,3,H,W)

    # Take SHAP values for the predicted class, squeeze to (3, H, W)
    sv = np.array(shap_values[pred_class]).squeeze(0)  # (3, H, W)

    # Aggregate across channels: mean absolute value per pixel → (H, W)
    sv_map = np.mean(np.abs(sv), axis=0)
    sv_map = (sv_map - sv_map.min()) / (sv_map.max() - sv_map.min() + 1e-8)

    # Signed map (red = helps prediction, blue = hurts prediction)
    sv_signed = np.mean(sv, axis=0)
    sv_signed_norm = sv_signed / (np.abs(sv_signed).max() + 1e-8)  # [-1, 1]

    # Create red-blue heatmap
    h, w  = sv_signed_norm.shape
    rgb   = np.zeros((h, w, 3), dtype=np.float32)
    pos   = np.clip(sv_signed_norm, 0, 1)
    neg   = np.clip(-sv_signed_norm, 0, 1)
    rgb[:, :, 0] = pos   # red channel: positive SHAP
    rgb[:, :, 2] = neg   # blue channel: negative SHAP

    # Blend with original image
    rgb_float  = _tensor_to_rgb_float(image_tensor)
    overlay    = cv2.addWeighted(rgb_float, 0.5, rgb, 0.5, 0)
    overlay    = np.clip(overlay, 0, 1)

    return {
        "overlay_base64": _encode_overlay(overlay),
    }


# --------------- Unified entry point ---------------

def explain_prediction(model, image_tensor, pred_class, include_shap=False):
    """
    Run all explanation methods and return their results.

    Args:
        model         : EfficientNetClassifier in eval mode
        image_tensor  : (1, 3, 224, 224) tensor on device
        pred_class    : int, predicted class index
        include_shap  : bool — SHAP is slower (~2-3s), set False for fast inference

    Returns:
        dict with 'gradcam', 'gradcam_plus', and optionally 'shap'
    """
    result = {}

    try:
        result["gradcam"]      = run_gradcam(model, image_tensor, pred_class)
    except Exception as e:
        result["gradcam"]      = {"error": str(e)}

    try:
        result["gradcam_plus"] = run_gradcam_plus(model, image_tensor, pred_class)
    except Exception as e:
        result["gradcam_plus"] = {"error": str(e)}

    if include_shap:
        try:
            result["shap"]     = run_shap(model, image_tensor, pred_class)
        except Exception as e:
            result["shap"]     = {"error": str(e)}

    return result
