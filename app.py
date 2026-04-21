
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

    st.header("✍️ Text Input")

    user_text = st.text_area("Enter your symptoms:", height=150)

    if st.button("Analyze Text"):

        if user_text.strip() == "":
            st.warning("Please enter symptoms first.")
        else:
            with st.spinner("Analyzing symptoms..."):
                result = process_text(user_text)

            st.success("🧠 Analysis Result")
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

    st.header("📷 Image Analysis")

    img_file = st.file_uploader("Upload skin image", type=["jpg", "png", "jpeg"])

    if img_file is not None:

        if st.button("Analyze Image"):

            st.info("AI is analyzing image...")

            # ✅ FIX: dictionary return handle
            result = process_image_with_gradcam(img_file)

            label = result["label"]
            confidence = result["confidence"]
            cam_image = result["image"]
            info = result["disease_info"]

            #  ------------------------
            # prediction output
            # ------------------------
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2%}")

            #  ------------------------
            # disease info (NEW FEATURE)
            #------------------------
            st.markdown("### 🧠 Medical Explanation")

            st.write(f"**Disease Name:** {info['name']}")
            st.write(f"**Type:** {info['type']}")

            st.write("🧬 Symptoms:")
            st.write(info["symptoms"])

            st.write("💡 Advice:")
            st.write(info["advice"])

            # ------------------------
            # confidence warning system
            # ------------------------
            if confidence > 0.8:
                st.success("High confidence prediction ✅")
            elif confidence > 0.5:
                st.warning("Medium confidence ⚠️")
            else:
                st.error("Low confidence ❗ Please use clearer image")

            # =====================================================
            # Grad-CAM image
            # =====================================================
            st.image(cam_image, caption="Grad-CAM Visualization 🔥")

    if st.button("⬅ Back"):
        go_to_page("home")