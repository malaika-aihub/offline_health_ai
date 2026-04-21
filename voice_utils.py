

import re
import speech_recognition as sr

# ------------------------
# CLEAN TEXT
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------
# COMPLETE SKIN RULE BASE
# ------------------------
SKIN_RULES = [
    {
        "disease": "Allergic Dermatitis",
        "symptoms": ["itching", "rash", "redness", "swelling"],
        "confidence": 0.85,
        "advice": "Avoid allergens, use antihistamine cream, keep skin clean."
    },
    {
        "disease": "Eczema (Atopic Dermatitis)",
        "symptoms": ["itching", "dry skin", "red patches", "cracking"],
        "confidence": 0.90,
        "advice": "Moisturize regularly, avoid hot water, use mild steroid cream."
    },
    {
        "disease": "Fungal Infection (Tinea)",
        "symptoms": ["itching", "ring shaped rash", "scaling", "red patches"],
        "confidence": 0.88,
        "advice": "Keep area dry, use antifungal cream, avoid sharing clothes/towels."
    },
    {
        "disease": "Psoriasis",
        "symptoms": ["thick skin", "scaling", "red patches", "itching"],
        "confidence": 0.92,
        "advice": "Moisturize skin, avoid stress, consult dermatologist."
    },
    {
        "disease": "Acne Vulgaris",
        "symptoms": ["pimples", "oily skin", "whiteheads", "blackheads"],
        "confidence": 0.93,
        "advice": "Wash face twice daily, avoid oily food, use salicylic acid."
    },
    {
        "disease": "Heat Rash",
        "symptoms": ["itching", "small red bumps", "sweating", "irritation"],
        "confidence": 0.80,
        "advice": "Stay cool, wear loose clothes, keep skin dry."
    },

    # ------------------------ PRE-CANCER ------------------------
    {
        "disease": "Actinic Keratosis (Pre-cancer)",
        "symptoms": ["rough scaly patch", "sun damaged skin", "dry crusted lesion"],
        "confidence": 0.93,
        "advice": "⚠️ Pre-cancer warning: Consult dermatologist immediately."
    },
    {
        "disease": "Bowen’s Disease (Early Skin Cancer)",
        "symptoms": ["red scaly patch", "non healing lesion", "slow growing patch"],
        "confidence": 0.94,
        "advice": "⚠️ Early cancer: Medical evaluation required."
    },

    # ------------------------SKIN CANCERS ------------------------
    {
        "disease": "Basal Cell Carcinoma (BCC)",
        "symptoms": ["pearly bump", "non healing sore", "pink patch"],
        "confidence": 0.95,
        "advice": "🚨 Cancer: Dermatologist treatment required."
    },
    {
        "disease": "Squamous Cell Carcinoma (SCC)",
        "symptoms": ["red firm bump", "scaly patch", "bleeding wound"],
        "confidence": 0.95,
        "advice": "🚨 High risk cancer: Immediate medical attention needed."
    },
    {
        "disease": "Melanoma",
        "symptoms": ["irregular mole", "changing mole", "black patch", "asymmetry mole"],
        "confidence": 0.98,
        "advice": "🚨 EMERGENCY: Possible melanoma. Immediate doctor visit."
    },
    {
        "disease": "Merkel Cell Carcinoma",
        "symptoms": ["fast growing lump", "painless nodule", "red purple bump"],
        "confidence": 0.97,
        "advice": "🚨 Rare aggressive cancer: Immediate hospital required."
    }
]


# ------------------------
# RULE ENGINE
# ------------------------
def rule_engine(user_text):

    user_text = clean_text(user_text)

    best_match = None
    best_score = 0

    for rule in SKIN_RULES:

        match_count = 0

        for sym in rule["symptoms"]:

            score = 0

            # exact match (strong)
            if sym in user_text:
                score = 1

            # word match (weak)
            elif len(set(user_text.split()) & set(sym.split())) > 0:
                score = 0.5

            match_count += score

        # normalize score
        final_score = match_count / len(rule["symptoms"])

        if final_score > best_score:
            best_score = final_score
            best_match = rule

    if best_match is None or best_score == 0:
        return {
            "disease": "Unknown / Mild Skin Condition",
            "confidence": 0.40,
            "advice": "Monitor symptoms and consult doctor if needed."
        }

    return {
        "disease": best_match["disease"],
        "confidence": round(best_score, 2),
        "advice": best_match["advice"]
    }

#------------------------
# VOICE TO TEXT (MIC FUNCTION)
#------------------------
import speech_recognition as sr

def voice_to_text(audio_file):

    r = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)

        text = r.recognize_google(audio)
        return text

    except:
        return "Sorry, could not understand audio"


# ------------------------
# FORMAT OUTPUT
# ------------------------
def format_response(result):

    confidence = result["confidence"] * 100

    if confidence >= 85:
        level = "🔴 High Confidence"
    elif confidence >= 60:
        level = "🟠 Medium Confidence"
    else:
        level = "🟡 Low Confidence"

    return f"""
🧠 Disease Prediction: {result['disease']}

📊 Confidence: {confidence:.1f}% ({level})

💡 Medical Advice:
{result['advice']}
"""


# ------------------------
# MAIN PIPELINE (VOICE + RULE ENGINE)
#------------------------
def run_voice_pipeline():

    # 🎤 Step 1: Voice input
    text = voice_to_text()

    # 🧠 Step 2: Rule engine
    result = rule_engine(text)

    # 📊 Step 3: formatted output
    return format_response(result)


# ------------------------
# TEXT INPUT SUPPORT (optional)
#------------------------
def process_input(text):
    result = rule_engine(text)
    return format_response(result)