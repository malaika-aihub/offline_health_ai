import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv

# ------------------------
# LOAD ENV
# ------------------------


HF_TOKEN = st.secrets.get("HF_TOKEN") 


if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

#  ONLY GEMMA 4 MODEL (rule)
MODEL_NAME = "google/gemma-4-E2B-it"


# ------------------------
# CLEAN TEXT
# ------------------------
def clean_text(text):
    return text.lower().strip()


# ------------------------
# SKIN RULES (YOUR LOCAL AI BRAIN)
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

    # PRE-CANCER
    {
        "disease": "Actinic Keratosis (Pre-cancer)",
        "symptoms": ["rough scaly patch", "sun damaged skin", "dry crusted lesion"],
        "confidence": 0.93,
        "advice": "⚠️ Pre-cancer warning: Consult dermatologist immediately."
    },

    # SKIN CANCERS
    {
        "disease": "Basal Cell Carcinoma (BCC)",
        "symptoms": ["pearly bump", "non healing sore", "pink patch"],
        "confidence": 0.95,
        "advice": "🚨 Cancer: Dermatologist treatment required."
    },
    {
        "disease": "Melanoma",
        "symptoms": ["irregular mole", "changing mole", "black patch", "asymmetry mole"],
        "confidence": 0.98,
        "advice": "🚨 EMERGENCY: Possible melanoma. Immediate doctor visit."
    }
]


# ------------------------
# RULE ENGINE (LOCAL AI)
# ------------------------
def rule_engine(user_text):

    user_text = clean_text(user_text)

    best_match = None
    best_score = 0

    for rule in SKIN_RULES:
        match_score = 0

        for sym in rule["symptoms"]:
            if sym in user_text:
                match_score += 1

        score = match_score / len(rule["symptoms"])

        if score > best_score:
            best_score = score
            best_match = rule

    if best_match is None or best_score == 0:
        return None

    return {
        "disease": best_match["disease"],
        "confidence": best_score,
        "advice": best_match["advice"]
    }


# ------------------------
# FALLBACK RESPONSE
# ------------------------
def fallback_response(rule_result):

    if rule_result:
        return f"""
CAUSE:
{rule_result['disease']} (based on symptom matching)

SAFETY:
Monitor your condition. If it worsens, consult a dermatologist.

ADVICE:
{rule_result['advice']}

(Note: Response generated using local medical inference engine for guaranteed availability.)
"""

    return """
CAUSE:
Based on general medical patterns, your symptoms may indicate a mild condition.

SAFETY:
Monitor symptoms and consult doctor if needed.

ADVICE:
Stay hydrated, rest properly, and avoid self-medication.

(Note: Offline fallback response.)
"""


# ------------------------
# API CALL (GEMMA 4)
# ------------------------
def call_model(payload, user_input=None):

    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

    for attempt in range(5):

        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )

            print("\nMODEL:", MODEL_NAME)
            print("ATTEMPT:", attempt + 1)
            print("STATUS:", response.status_code)
            print("RAW RESPONSE:", response.text)

            if response.status_code != 200:
                time.sleep(5)
                continue

            result = response.json()

            if isinstance(result, dict) and "error" in result:
                return fallback_response(rule_engine(user_input))

            if isinstance(result, list):
                return result[0].get("generated_text", str(result))

        except:
            time.sleep(5)

    return fallback_response(rule_engine(user_input))


# ------------------------
# MAIN FUNCTION (HYBRID BRAIN)
# ------------------------
def process_text(user_input):

    # Step 1: try rule engine first
    rule_result = rule_engine(user_input)

    prompt = f"""
You are a medical AI assistant.

User symptoms:
{user_input}

Give:
CAUSE:
SAFETY:
ADVICE:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7
        }
    }

    # Step 2: try Gemma API
    api_result = call_model(payload, user_input)

    # Step 3: if API fails → fallback handled inside call_model
    return api_result