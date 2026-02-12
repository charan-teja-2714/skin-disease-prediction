"""
Simulated doctor chat responses for demo purposes.
Generates contextually appropriate replies based on the detected condition
and severity tier from the prediction.
"""

import random

# Condition-aware full names for natural conversation
CONDITION_NAMES = {
    "MEL": "melanoma",
    "BCC": "basal cell carcinoma",
    "AKIEC": "actinic keratosis",
    "NV": "melanocytic nevus",
    "BKL": "benign keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular lesion",
}

GREETINGS = [
    "Hello! Thank you for reaching out. I've reviewed the screening results from your upload.",
    "Hi there. I can see the analysis report for your skin lesion. Let me share my thoughts.",
    "Welcome! I have your screening report in front of me. Let's discuss this.",
]

SEVERITY_RESPONSES = {
    "high": [
        "Based on the screening, the lesion shows features that warrant closer examination. "
        "I would strongly recommend an in-person visit so I can perform a dermoscopic evaluation. "
        "Please bring any previous images of the lesion if available.",
        "The analysis indicates some concerning characteristics. While the AI screening is a "
        "useful tool, I'd like to examine this personally. Can you visit the clinic soon?",
    ],
    "low": [
        "The screening results look reassuring overall. The lesion appears to have benign "
        "characteristics. That said, I'd still recommend keeping an eye on it for any changes.",
        "Good news — the analysis suggests this is likely benign. I'd recommend routine "
        "monitoring. Please re-check if you notice changes in size, shape, or color.",
    ],
}

CONDITION_ADVICE = {
    "MEL": "Given the melanoma indication, early evaluation is critical. Time is an important factor here.",
    "BCC": "Basal cell carcinoma, if confirmed, is very treatable when caught early. An in-person biopsy would give us a definitive answer.",
    "AKIEC": "Actinic keratoses can progress if left untreated. I'd like to assess whether treatment is needed.",
    "NV": "Moles are usually harmless, but monitoring for asymmetry or color changes is important.",
    "BKL": "Benign keratoses are non-cancerous. If it's causing discomfort or concern, we can discuss removal options.",
    "DF": "Dermatofibromas are typically harmless. Unless it's growing or painful, observation is usually sufficient.",
    "VASC": "Vascular lesions vary widely. I'd like to examine this to determine the specific type and any treatment needed.",
}

FOLLOWUPS = [
    "Could you tell me more about when you first noticed this lesion?",
    "Has the lesion changed in size, shape, or color recently?",
    "Do you have any family history of skin cancer or melanoma?",
    "Are you experiencing any itching, bleeding, or pain in the affected area?",
    "How much sun exposure do you typically get? Do you use sunscreen regularly?",
    "Do you have any other lesions or moles that have been concerning you?",
    "Are you currently on any medications or topical treatments?",
    "I'd recommend taking photos of the lesion monthly to track any changes. Would you like guidance on how to do this properly?",
    "Please make sure to protect the area from direct sun exposure in the meantime.",
    "If you notice any sudden changes before our next appointment, please don't hesitate to reach out immediately.",
]


def generate_doctor_reply(message, disease_code="", severity_tier="low", is_first=False):
    """
    Generate a simulated doctor response.

    Args:
        message: The patient's message text
        disease_code: The predicted disease code (e.g., "MEL")
        severity_tier: "high" or "low"
        is_first: Whether this is the first message in the conversation

    Returns:
        A doctor reply string.
    """
    if is_first:
        greeting = random.choice(GREETINGS)
        tier = "high" if severity_tier == "high" else "low"
        severity_reply = random.choice(SEVERITY_RESPONSES[tier])
        condition_note = CONDITION_ADVICE.get(disease_code, "")
        parts = [greeting, severity_reply]
        if condition_note:
            parts.append(condition_note)
        return " ".join(parts)

    # For follow-up messages, pick a contextual follow-up
    return random.choice(FOLLOWUPS)
