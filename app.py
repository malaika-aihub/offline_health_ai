
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import os
from text_utils import process_text
from voice_utils import voice_to_text
from image_utils import process_image_with_gradcam


#------------------------
# PAGE CONFIG
#------------------------
st.set_page_config(page_title="Offline Health AI", layout="centered")

# ------------------------
# SESSION STATE
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(p):
    st.session_state.page = p

#------------------------
# HOME PAGE
#------------------------
if st.session_state.page == "home":
    st.title("🧠 Offline Health AI Assistant")

    st.markdown("""
### 📌 Multi-Modal Medical AI System

This system supports:
- ✍️ Text symptoms
- 🎤 Voice input
- 📷 Skin image analysis

---

### 📸 IMPORTANT
For best results, upload a **clear skin image** for visual diagnosis.
""")

    st.markdown("### 🚀 Choose input method:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✍️ Text"):
            go_to_page("text")

    with col2:
        if st.button("🎤 Voice"):
            go_to_page("voice")

    with col3:
        if st.button("📷 Photo"):
            go_to_page("photo")

# ------------------------
# TEXT PAGE
# ------------------------
if st.session_state.page == "text":

    st.markdown("""
    <style>
    div[data-baseweb="textarea"] textarea {
        resize: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("✍️ Symptom Analysis (Gemma 4 AI)")

    user_text = st.text_area("Enter your symptoms:", height=150)

    if st.button("Analyze Text"):

        if user_text.strip() == "":
            st.warning("⚠️ Please enter symptoms first.")
        else:
            with st.spinner("🧠 Gemma 4 is analyzing your symptoms..."):
                
                # AI call
                result = process_text(user_text)

            st.success("🧠 AI Analysis Result")

            st.markdown("### 📋 Report")
            st.write(result)

    if st.button("⬅ Back"):
        go_to_page("home")

# ------------------------
# VOICE PAGE
#  ------------------------
import speech_recognition as sr

if st.session_state.page == "voice":

    st.header("🎤 Voice Input - AI Doctor")

    audio_file = st.audio_input("Speak your symptoms")

    if audio_file is not None:

        if st.button("Analyze Voice"):

            r = sr.Recognizer()

            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)

            try:
                text = r.recognize_google(audio)

            except:
                text = "Sorry, could not understand audio"

            st.success("🧾 Converted Text:")
            st.write(text)

            st.success("🧠 Medical Analysis:")
            result = process_text(text)
            st.write(result)


    if st.button("⬅ Back"):
        go_to_page("home")

#  ------------------------
# PHOTO PAGE
#------------------------
if st.session_state.page == "photo":

    st.markdown("## 📷 AI Skin Analyzer")

    # 🌍 Language selector
    language = st.selectbox(
        "🌍 Select Language",
        ["English", "Urdu", "Hindi", "Arabic", "French"]
    )

    lang_map = {
        "English": "en",
        "Urdu": "ur",
        "Hindi": "hi",
        "Arabic": "ar",
        "French": "fr"
    }

    selected_lang = lang_map[language]

    img_file = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])

    if img_file is not None:

        st.image(img_file, width=500)

        if st.button("🔍 Analyze Image"):

            with st.spinner("🧠 AI is analyzing..."):

                result = process_image_with_gradcam(img_file, selected_lang)

            label = result["label"]
            confidence = result["confidence"]
            cam_image = result["image"]
            info = result["disease_info"]

            st.success("✅ Analysis Complete")

            # 🎯 Prediction
            st.markdown("### 🧾 Prediction")
            st.write(label)
            st.progress(confidence)

            # 🧠 Info
            st.markdown("### 🧠 Medical Details")
            st.write(f"**Disease:** {info['name']}")
            st.write(f"**Type:** {info['type']}")

            st.markdown("**Symptoms:**")
            for s in info["symptoms"]:
                st.write(f"- {s}")

            st.markdown("**Advice:**")
            st.info(info["advice"])

            # Confidence
            if confidence > 0.8:
                st.success("High Confidence ✅")
            elif confidence > 0.5:
                st.warning("Medium Confidence ⚠️")
            else:
                st.error("Low Confidence ❗")

            # GradCAM
            st.markdown("### 🔥 AI Focus Area")
            st.image(cam_image)
    if st.button("⬅ Back"):
        go_to_page("home")
















