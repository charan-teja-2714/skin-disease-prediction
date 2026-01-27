import cv2
import numpy as np

# ---------- A: Asymmetry ----------
def asymmetry_score(mask):
    h, w = mask.shape
    left = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])

    diff = np.abs(left[:, :right.shape[1]] - right)
    score = np.sum(diff) / np.sum(mask) if np.sum(mask) > 0 else 0
    return round(score, 3)

# ---------- B: Border Irregularity ----------
def border_irregularity(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)

    if area == 0 or perimeter == 0:
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    irregularity = 1 - circularity
    return round(max(0, min(1, irregularity)), 3)

# ---------- C: Color Variation ----------
def color_variation(image, mask, k=3):
    lesion_pixels = image[mask > 0]
    if len(lesion_pixels) < 10:
        return 0.5

    lesion_pixels = lesion_pixels.reshape(-1, 3)
    lesion_pixels = np.float32(lesion_pixels)

    try:
        _, labels, _ = cv2.kmeans(
            lesion_pixels, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10, cv2.KMEANS_RANDOM_CENTERS
        )
        unique_colors = len(np.unique(labels))
        # Normalize to 0-1 scale
        return round(min(unique_colors / k, 1.0), 3)
    except:
        return 0.5

# ---------- D: Diameter ----------
def diameter_mm(mask, pixel_to_mm=0.1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter_px = radius * 2
    diameter_mm = diameter_px * pixel_to_mm

    return round(diameter_mm, 2)

# ---------- E: Evolution ----------
def evolution_risk(asymmetry, border, color, diameter):
    """Calculate evolution risk based on other ABCDE factors"""
    # Simple heuristic: higher values in other factors suggest evolution risk
    risk_score = (asymmetry * 0.3 + border * 0.3 + color * 0.2 + min(diameter/10, 1) * 0.2)
    return round(min(risk_score, 1.0), 3)

class ABCDEAnalyzer:
    """ABCDE Analysis for skin lesions"""
    
    def __init__(self, pixel_to_mm=0.1):
        self.pixel_to_mm = pixel_to_mm
    
    def analyze_lesion(self, image, mask):
        """Perform complete ABCDE analysis"""
        # Ensure mask is binary
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Calculate each ABCDE component
        asymmetry = asymmetry_score(mask)
        border = border_irregularity(mask)
        color = color_variation(image, mask)
        diameter = diameter_mm(mask, self.pixel_to_mm)
        evolution = evolution_risk(asymmetry, border, color, diameter)
        
        # Generate clinical interpretation
        interpretation = self._generate_interpretation(asymmetry, border, color, diameter, evolution)
        
        return {
            'asymmetry_score': asymmetry,
            'border_irregularity': border,
            'color_variation': color,
            'diameter_mm': diameter,
            'evolution_risk': evolution,
            'clinical_interpretation': interpretation,
            'overall_abcde_score': (asymmetry + border + color + evolution + min(diameter/10, 1)) / 5
        }
    
    def _generate_interpretation(self, asymmetry, border, color, diameter, evolution):
        """Generate clinical interpretation of ABCDE scores"""
        interpretation = []
        
        # Asymmetry
        if asymmetry > 0.5:
            interpretation.append("High asymmetry detected - lesion is not symmetrical")
        elif asymmetry > 0.3:
            interpretation.append("Moderate asymmetry present")
        else:
            interpretation.append("Lesion appears relatively symmetrical")
        
        # Border
        if border > 0.6:
            interpretation.append("Highly irregular borders detected")
        elif border > 0.4:
            interpretation.append("Moderately irregular borders")
        else:
            interpretation.append("Borders appear relatively regular")
        
        # Color
        if color > 0.7:
            interpretation.append("High color variation - multiple colors present")
        elif color > 0.4:
            interpretation.append("Moderate color variation")
        else:
            interpretation.append("Relatively uniform coloration")
        
        # Diameter
        if diameter > 6:
            interpretation.append(f"Large diameter ({diameter:.1f}mm) - exceeds 6mm threshold")
        else:
            interpretation.append(f"Diameter within normal range ({diameter:.1f}mm)")
        
        # Evolution
        if evolution > 0.6:
            interpretation.append("High evolution risk - multiple concerning features")
        elif evolution > 0.4:
            interpretation.append("Moderate evolution risk")
        else:
            interpretation.append("Low evolution risk")
        
        return interpretation
