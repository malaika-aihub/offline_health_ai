
# import tempfile
# import os
# import re
# import whisper

# # =====================================================
# # FFmpeg PATH FIX
# # =====================================================
# os.environ["PATH"] += os.pathsep + r"C:\Users\MASTER\Downloads\ffmpeg\bin"

# # =====================================================
# # LOAD WHISPER MODEL
# # =====================================================
# model = whisper.load_model("base")

# # =====================================================
# # KEYWORDS (skin detection)
# # =====================================================
# skin_keywords = [
#     "itching", "rash", "spots", "lesion",
#     "skin", "mole", "burning", "acne"
# ]

# def is_skin_related(text):
#     text = text.lower()
#     return any(word in text for word in skin_keywords)

# # =====================================================
# # SPELLING FIX MAP
# # =====================================================
# replacements = {
#     "eaching": "itching",
#     "eating": "itching",
#     "burningg": "burning",
#     "rassh": "rash",
#     "rashh": "rash",
#     "precious": "",
#     "pleaes": "please",
#     "plese": "please"
# }

# # =====================================================
# # CLEAN TEXT
# # =====================================================
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s]", "", text)

#     for wrong, correct in replacements.items():
#         text = text.replace(wrong, correct)

#     return text.strip()

# # =====================================================
# # GEMMA FUNCTION (SAFE MODE)
# # =====================================================
# def call_gemma(text):

#     if len(text.split()) < 3:
#         return "Uncertain: please provide clear symptoms or upload image."

#     if not is_skin_related(text):
#         return "Uncertain: not a skin-related symptom."

#     prompt = f"""
# You are a STRICT skin health assistant.

# User symptoms: {text}

# Return ONLY:
# CAUSE:
# SAFETY:
# ADVICE:
# IMAGE:
# """

    
#     # currently placeholder output
#     response = f"""CAUSE: Skin irritation
# SAFETY:
# - Avoid scratching
# - Keep skin clean
# - Use mild soap 
# - ADVICE: Maintain hygiene  
# - IMAGE: Upload image for better analysis"""

#     return response

# # =====================================================
# # VOICE PROCESSING
# # =====================================================
# def process_voice(audio_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     try:
#         audio_file.seek(0)

#         # Whisper STT
#         result = model.transcribe(tmp_path)
#         raw_text = result["text"].strip()

#         # Clean text
#         cleaned_text = clean_text(raw_text)

#         # Gemma response
#         gemma_output = call_gemma(cleaned_text)

#         return cleaned_text, gemma_output

#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)



import tempfile
import os
import re
import whisper
import torch


# =====================================================
# FFmpeg PATH FIX
# =====================================================
os.environ["PATH"] += os.pathsep + r"C:\Users\MASTER\Downloads\ffmpeg\bin"

# =====================================================
# WHISPER LOAD
# =====================================================
model = whisper.load_model("base")

# =====================================================
# IMPORT EXISTING GEMMA PIPELINE (NO NEW MODEL LOAD)
# =====================================================
from text_utils import process_text


# =====================================================
# SPELLING FIX MAP
# =====================================================
replacements = {
    "eaching": "itching",
    "eating": "itching",
    "burningg": "burning",
    "rassh": "rash",
    "rashh": "rash",
    "pleaes": "please",
    "plese": "please"
}

# =====================================================
# CLEAN TEXT
# =====================================================
def clean_text(text):
    text = text.lower()

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# VOICE PROCESSING
# =====================================================
def process_voice(audio_file):

    tmp_path = None

    try:
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        audio_file.seek(0)

        # Whisper transcription
        result = model.transcribe(tmp_path)
        raw_text = result.get("text", "").strip()

        print("🧾 Whisper Output:", raw_text)

        # Clean text
        cleaned_text = clean_text(raw_text)

        print("🧹 Cleaned Text:", cleaned_text)

        # =====================================================
        # SEND TO EXISTING GEMMA PIPELINE (text_utils.py)
        # =====================================================
        gemma_response = process_text(cleaned_text)

        return cleaned_text, gemma_response

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)











# import tempfile
# import os
# import re
# import whisper
# import pyttsx3

# # =====================================================
# # FFmpeg PATH FIX
# # =====================================================
# os.environ["PATH"] += os.pathsep + r"C:\Users\MASTER\Downloads\ffmpeg\bin"

# # =====================================================
# # WHISPER (LOAD ONCE ONLY)
# # =====================================================
# model = whisper.load_model("base")

# # =====================================================
# # GEMMA PIPELINE
# # =====================================================
# from text_utils import process_text

# # =====================================================
# # SINGLE TTS ENGINE (IMPORTANT FIX)
# # =====================================================
# tts_engine = pyttsx3.init()
# tts_engine.setProperty('rate', 170)

# # =====================================================
# # SPELL FIX
# # =====================================================
# replacements = {
#     "eaching": "itching",
#     "eating": "itching",
#     "burningg": "burning",
#     "rassh": "rash",
#     "rashh": "rash",
#     "pleaes": "please",
#     "plese": "please"
# }

# # =====================================================
# # CLEAN TEXT
# # =====================================================
# def clean_text(text):
#     text = text.lower()

#     for wrong, correct in replacements.items():
#         text = text.replace(wrong, correct)

#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     return text

# # =====================================================
# # SPEAK (STABLE FIX)
# # =====================================================
# def speak(text):
#     try:
#         if not text:
#             return

#         tts_engine.stop()          # 🔥 important reset
#         tts_engine.say(text)
#         tts_engine.runAndWait()

#     except Exception as e:
#         print("TTS Error:", e)

# # =====================================================
# # MAIN PIPELINE
# # =====================================================
# def process_voice(audio_file, voice_only=True):

#     tmp_path = None

#     try:
#         # -----------------------------
#         # SAVE AUDIO SAFELY
#         # -----------------------------
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(audio_file.read())
#             tmp_path = tmp.name

#         audio_file.seek(0)

#         # -----------------------------
#         # WHISPER TRANSCRIPTION
#         # -----------------------------
#         result = model.transcribe(tmp_path)
#         raw_text = result.get("text", "").strip()

#         print("🧾 Whisper:", raw_text)

#         # -----------------------------
#         # CLEAN
#         # -----------------------------
#         cleaned_text = clean_text(raw_text)

#         print("🧹 Clean:", cleaned_text)

#         # -----------------------------
#         # GEMMA OUTPUT
#         # -----------------------------
#         gemma_response = process_text(cleaned_text)

#         print("🤖 Gemma OK")

#         # -----------------------------
#         # VOICE OUTPUT (FIXED)
#         # -----------------------------
#         speak(gemma_response)

#         # -----------------------------
#         # RETURN CONTROL
#         # -----------------------------
#         if voice_only:
#             return None, None
#         else:
#             return cleaned_text, gemma_response

#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)