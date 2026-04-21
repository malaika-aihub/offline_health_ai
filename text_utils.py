
import re

#------------------------
# CLEAN TEXT
# ------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------
# RULE DATABASE (CAUSE + SAFETY + ADVICE)
#------------------------
MEDICAL_RULES = [

    # ------------------------ COMMON SKIN ISSUES ------------------------

    {
        "keywords": ["itching", "rash", "redness", "swelling"],
        "cause": "Allergic reaction or irritant exposure",
        "safety": "Mild condition but monitor for worsening symptoms",
        "advice": "Avoid allergens, use antihistamine cream, keep skin clean."
    },
    {
        "keywords": ["dry skin", "itching", "cracking", "red patches"],
        "cause": "Eczema due to skin barrier weakness",
        "safety": "Chronic but non-life-threatening condition",
        "advice": "Moisturize regularly, avoid hot water, use mild steroid cream."
    },
    {
        "keywords": ["ring shaped rash", "scaling", "itching"],
        "cause": "Fungal infection due to moisture and poor hygiene",
        "safety": "Contagious but treatable",
        "advice": "Keep area dry, use antifungal cream, avoid sharing clothes."
    },
    {
        "keywords": ["pimples", "oily skin", "whiteheads", "blackheads"],
        "cause": "Blocked pores due to oil and bacteria",
        "safety": "Harmless but may cause scarring if untreated",
        "advice": "Wash face twice daily, avoid oily food, use salicylic acid."
    },

    #------------------------ PRE-CANCER CONDITIONS ------------------------

    {
        "keywords": ["rough scaly patch", "sun damaged skin", "thickened skin"],
        "cause": "UV radiation damage leading to precancerous changes",
        "safety": "⚠️ PRE-CANCER: Needs medical attention",
        "advice": "Consult dermatologist, avoid sun exposure, use sunscreen."
    },
    {
        "keywords": ["red scaly patch", "non healing lesion", "slow growing patch"],
        "cause": "Early malignant transformation of skin cells",
        "safety": "⚠️ HIGH RISK PRE-CANCER",
        "advice": "Medical evaluation required immediately."
    },

    # ------------------------SKIN CANCERS ------------------------

    {
        "keywords": ["pearly bump", "non healing sore", "pink patch"],
        "cause": "Basal cell carcinoma due to UV damage",
        "safety": "🚨 SKIN CANCER (slow growing)",
        "advice": "Dermatologist treatment required."
    },
    {
        "keywords": ["red firm bump", "scaly patch", "bleeding wound"],
        "cause": "Squamous cell carcinoma due to sun exposure",
        "safety": "🚨 HIGH RISK CANCER",
        "advice": "Immediate medical attention required."
    },
    {
        "keywords": ["irregular mole", "changing mole", "black patch", "asymmetry mole"],
        "cause": "Melanoma due to abnormal melanocyte growth",
        "safety": "🚨 DANGEROUS SKIN CANCER",
        "advice": "Emergency dermatologist consultation needed."
    },
    {
        "keywords": ["fast growing lump", "painless nodule", "red purple bump"],
        "cause": "Merkel cell carcinoma (rare aggressive cancer)",
        "safety": "🚨 VERY AGGRESSIVE CANCER",
        "advice": "Immediate hospital visit required."
    }
]


# ------------------------
# RULE ENGINE
# ------------------------
def rule_engine(user_text):

    user_text = clean_text(user_text)

    best_match = None
    best_score = 0

    for rule in MEDICAL_RULES:

        match_count = 0

        for kw in rule["keywords"]:
            if kw in user_text:
                match_count += 1

        score = match_count / len(rule["keywords"])

        if score > best_score:
            best_score = score
            best_match = rule

    if best_match is None or best_score == 0:
        return {
            "cause": "Unknown or mild skin irritation",
            "safety": "Low risk condition",
            "advice": "Monitor symptoms. Consult doctor if it persists."
        }

    return {
        "cause": best_match["cause"],
        "safety": best_match["safety"],
        "advice": best_match["advice"]
    }


# ------------------------
# FORMAT OUTPUT (LLM STYLE RESPONSE)
# ------------------------
def process_text(user_input):

    result = rule_engine(user_input)

    response = f"""
CAUSE :  {result['cause']}
SAFETY :  {result['safety']}
ADVICE :  {result['advice']}
"""

    return response.strip()