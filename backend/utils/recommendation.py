"""
Comprehensive clinical recommendation engine.

Generates detailed, multi-tier recommendations based on:
- Disease classification and severity
- Prediction confidence and uncertainty
- ABCDE morphological analysis
- Skin tone (for fairness transparency)
- Image quality metrics
- Longitudinal evolution (when available)
"""

# Disease severity tiers based on clinical risk
DISEASE_SEVERITY = {
    "MEL":   {"tier": "high",   "label": "Melanoma",                  "urgency": 5},
    "BCC":   {"tier": "high",   "label": "Basal Cell Carcinoma",      "urgency": 4},
    "AKIEC": {"tier": "high",   "label": "Actinic Keratosis / Bowen", "urgency": 4},
    "BKL":   {"tier": "low",    "label": "Benign Keratosis",          "urgency": 1},
    "NV":    {"tier": "low",    "label": "Melanocytic Nevus",         "urgency": 1},
    "DF":    {"tier": "low",    "label": "Dermatofibroma",            "urgency": 1},
    "VASC":  {"tier": "low",    "label": "Vascular Lesion",           "urgency": 2},
}

# Recommendation type definitions
RECOMMENDATIONS = {
    "IMMEDIATE_CONSULTATION": {
        "level": "critical",
        "title": "Immediate Dermatologist Consultation Required",
        "color": "#dc2626",
        "icon": "critical",
        "description": (
            "The analysis indicates features consistent with a potentially serious "
            "skin condition. Please consult a dermatologist as soon as possible."
        ),
    },
    "HIGH_RISK_ALERT": {
        "level": "high",
        "title": "High-Risk Alert",
        "color": "#ea580c",
        "icon": "warning",
        "description": (
            "Multiple concerning features have been detected. A professional "
            "evaluation is strongly recommended within the next few days."
        ),
    },
    "MONITOR_CLOSELY": {
        "level": "moderate",
        "title": "Monitor Closely for Changes",
        "color": "#d97706",
        "icon": "monitor",
        "description": (
            "Some atypical features were detected. Monitor the lesion for any "
            "changes in size, shape, or color, and consult a dermatologist if "
            "changes occur."
        ),
    },
    "ROUTINE_MONITORING": {
        "level": "low",
        "title": "Routine Monitoring Suggested",
        "color": "#16a34a",
        "icon": "check",
        "description": (
            "The lesion appears benign with no immediately concerning features. "
            "Continue routine skin checks and annual dermatological exams."
        ),
    },
    "POOR_IMAGE_QUALITY": {
        "level": "warning",
        "title": "Re-upload Image — Poor Quality Detected",
        "color": "#7c3aed",
        "icon": "upload",
        "description": (
            "The uploaded image may be too blurry, too dark, or too small for "
            "reliable analysis. Please re-upload a higher-quality, well-lit, "
            "in-focus image of the lesion."
        ),
    },
    "UNCERTAIN_PREDICTION": {
        "level": "moderate",
        "title": "Uncertain Prediction — Expert Review Required",
        "color": "#9333ea",
        "icon": "uncertain",
        "description": (
            "The AI model's prediction has high uncertainty. The result may not "
            "be reliable. Please consult a dermatologist for a definitive diagnosis."
        ),
    },
    "EVOLUTION_FOLLOWUP": {
        "level": "moderate",
        "title": "Follow-up Suggested — Lesion Evolution Detected",
        "color": "#0284c7",
        "icon": "evolution",
        "description": (
            "Changes in the lesion have been detected compared to the previous "
            "image. A follow-up consultation is recommended to evaluate progression."
        ),
    },
    "NO_LESION_DETECTED": {
        "level": "info",
        "title": "No Skin Lesion Detected",
        "color": "#6b7280",
        "icon": "info",
        "description": (
            "The system could not detect a skin lesion in this image. "
            "Please upload a clear, close-up photo of the skin area you "
            "want analyzed. The image should show a visible mole, spot, or lesion."
        ),
    },
}


def generate_recommendation(
    disease,
    confidence,
    uncertainty,
    abcde_scores=None,
    skin_tone=None,
    image_quality=None,
    evolution_data=None,
):
    """
    Generate a comprehensive clinical recommendation.

    Returns a dict with:
      - primary_recommendation: the main recommendation dict
      - secondary_recommendations: list of additional advisories
      - risk_factors: list of identified risk factors
      - confidence_assessment: human-readable confidence statement
      - skin_tone_note: fairness transparency note (if applicable)
    """
    severity = DISEASE_SEVERITY.get(disease, DISEASE_SEVERITY["NV"])
    risk_factors = []
    secondary = []

    # ---- 1. Image quality gate ----
    if image_quality and not image_quality.get("acceptable", True):
        reasons = image_quality.get("issues", [])
        primary = dict(RECOMMENDATIONS["POOR_IMAGE_QUALITY"])
        primary["details"] = (
            "Issues detected: " + ", ".join(reasons) + ". "
            "Results below may not be reliable."
        )
        # Still return partial results but flag clearly
        secondary.append({
            "type": "quality_warning",
            "message": "Analysis was performed but results should be treated with caution.",
        })

        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # ---- 2. High uncertainty gate ----
    if uncertainty > 0.15:
        primary = dict(RECOMMENDATIONS["UNCERTAIN_PREDICTION"])
        primary["details"] = (
            f"Model uncertainty is {uncertainty:.1%}, which exceeds the reliability "
            f"threshold. The predicted class '{severity['label']}' may not be accurate."
        )
        risk_factors.append("High model uncertainty")
        if severity["tier"] == "high":
            secondary.append({
                "type": "severity_note",
                "message": (
                    f"Although uncertain, the predicted condition "
                    f"({severity['label']}) is clinically serious. "
                    f"Err on the side of caution."
                ),
            })
        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # ---- 3. Evolution-triggered recommendation ----
    if evolution_data and evolution_data.get("alert"):
        primary = dict(RECOMMENDATIONS["EVOLUTION_FOLLOWUP"])
        changes = []
        if evolution_data.get("area_change_pct", 0) > 20:
            changes.append(f"area increased by {evolution_data['area_change_pct']:.1f}%")
        if evolution_data.get("diameter_change_mm", 0) > 2:
            changes.append(f"diameter grew by {evolution_data['diameter_change_mm']:.1f}mm")
        if evolution_data.get("color_change", 0) >= 2:
            changes.append("significant color variation change")
        primary["details"] = "Detected changes: " + (", ".join(changes) if changes else "multiple morphological changes") + "."
        risk_factors.append("Lesion evolution detected")

        # If also high severity, escalate
        if severity["tier"] == "high":
            primary = dict(RECOMMENDATIONS["IMMEDIATE_CONSULTATION"])
            primary["details"] = (
                f"Lesion evolution detected AND predicted condition is "
                f"{severity['label']}. Immediate evaluation is critical."
            )
            risk_factors.append(f"High-severity condition: {severity['label']}")

        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # ---- 4. Disease-severity + confidence driven recommendation ----

    # Collect ABCDE risk factors
    abcde_risk_count = 0
    if abcde_scores:
        if abcde_scores.get("asymmetry_score", 0) > 0.5:
            risk_factors.append("High asymmetry")
            abcde_risk_count += 1
        elif abcde_scores.get("asymmetry_score", 0) > 0.3:
            abcde_risk_count += 0.5

        if abcde_scores.get("border_irregularity", 0) > 0.6:
            risk_factors.append("Irregular borders")
            abcde_risk_count += 1
        elif abcde_scores.get("border_irregularity", 0) > 0.4:
            abcde_risk_count += 0.5

        if abcde_scores.get("color_variation", 0) > 0.7:
            risk_factors.append("High color variation")
            abcde_risk_count += 1

        if abcde_scores.get("diameter_mm", 0) > 6:
            risk_factors.append(f"Large diameter ({abcde_scores['diameter_mm']:.1f}mm)")
            abcde_risk_count += 1

        if abcde_scores.get("evolution_risk", 0) > 0.6:
            risk_factors.append("High evolution risk score")
            abcde_risk_count += 1

    # 4a. Immediate consultation: high-severity disease + confident
    if severity["tier"] == "high" and confidence > 0.5:
        primary = dict(RECOMMENDATIONS["IMMEDIATE_CONSULTATION"])
        primary["details"] = (
            f"Predicted condition: {severity['label']} "
            f"(confidence: {confidence:.1%}). "
            f"This condition requires professional evaluation."
        )
        risk_factors.append(f"High-severity condition: {severity['label']}")
        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # 4b. High-risk alert: moderate confidence on high-severity OR many ABCDE flags
    if (severity["tier"] == "high" and confidence > 0.3) or abcde_risk_count >= 3:
        primary = dict(RECOMMENDATIONS["HIGH_RISK_ALERT"])
        detail_parts = []
        if severity["tier"] == "high":
            detail_parts.append(
                f"Predicted condition: {severity['label']} "
                f"(confidence: {confidence:.1%})"
            )
        if abcde_risk_count >= 3:
            detail_parts.append(
                f"{int(abcde_risk_count)} ABCDE risk factors identified"
            )
        primary["details"] = ". ".join(detail_parts) + "."
        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # 4c. Monitor closely: moderate uncertainty OR some ABCDE concerns
    if uncertainty > 0.08 or abcde_risk_count >= 2:
        primary = dict(RECOMMENDATIONS["MONITOR_CLOSELY"])
        detail_parts = []
        if uncertainty > 0.08:
            detail_parts.append(f"Moderate model uncertainty ({uncertainty:.1%})")
        if abcde_risk_count >= 2:
            detail_parts.append(f"{int(abcde_risk_count)} morphological risk factors")
        primary["details"] = ". ".join(detail_parts) + "."
        return _build_response(
            primary, secondary, risk_factors,
            confidence, uncertainty, disease, severity, skin_tone
        )

    # 4d. Routine monitoring: low-risk, confident, few ABCDE concerns
    primary = dict(RECOMMENDATIONS["ROUTINE_MONITORING"])
    primary["details"] = (
        f"Predicted condition: {severity['label']} "
        f"(confidence: {confidence:.1%}). "
        f"No major risk factors identified."
    )
    return _build_response(
        primary, secondary, risk_factors,
        confidence, uncertainty, disease, severity, skin_tone
    )


def _build_response(
    primary, secondary, risk_factors,
    confidence, uncertainty, disease, severity, skin_tone
):
    """Assemble the final recommendation response."""
    # Confidence assessment
    if confidence > 0.8:
        confidence_text = "High confidence"
    elif confidence > 0.5:
        confidence_text = "Moderate confidence"
    else:
        confidence_text = "Low confidence"

    result = {
        "primary_recommendation": primary,
        "secondary_recommendations": secondary,
        "risk_factors": risk_factors,
        "confidence_assessment": confidence_text,
        "disease_label": severity["label"],
        "disease_code": disease,
        "severity_tier": severity["tier"],
    }

    if skin_tone:
        result["skin_tone_note"] = (
            f"Detected skin tone: {skin_tone}. Model performance may vary across "
            f"skin tones. If results seem inconsistent, professional evaluation "
            f"is recommended."
        )

    return result
