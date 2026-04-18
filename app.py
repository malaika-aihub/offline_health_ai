import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from text_utils import process_text
from voice_utils import process_voice
from image_utils import process_image_with_gradcam
import pandas as pd
import os

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("data/raw/HAM10000_metadata.csv")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Offline Health AI", layout="centered")

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(p):
    st.session_state.page = p

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":
    st.title("Offline Health AI Assistant : )")
    st.markdown("""
### 📌 Welcome to Smart Medical AI System

This system supports **Text, Voice, and Image (Photo) input** for disease analysis.

---

### 📸 🔥 IMPORTANT NOTE 

👉 For **best and most accurate results**, please upload a **clear skin image**.

🖼️ Image-based analysis provides:
- Higher diagnostic accuracy
- Visual pattern detection
- More reliable medical prediction""")
    st.markdown("### 🚀 Choose your input method:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Text"):
            go_to_page("text")
    with col2:
        if st.button("Voice"):
            go_to_page("voice")
    with col3:
        if st.button("Photo"):
            go_to_page("photo")

# =====================================================
# TEXT PAGE
# =====================================================
if st.session_state.page == "text":
    st.header("Text Input")
    st.markdown("### Enter your symptoms here:")
    st.markdown("""
    <style>
    textarea {
        resize: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    user_text = st.text_area("", height=150)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result = process_text(user_text)
        st.success(result)

# =====================================================
# VOICE PAGE
# =====================================================
if st.session_state.page == "voice":

    st.header("🎤 Voice Input - AI Doctor")

    audio_file = st.audio_input("Speak your symptoms")

    if audio_file is not None:

        if st.button("Analyze Voice"):

            import time
            time.sleep(0.3)

            text, gemma_result = process_voice(audio_file)

            st.success("🧾 Converted Text:")
            st.write(text)

            st.info("🤖 Gemma Medical Analysis:")
            st.write(gemma_result)
                
# =====================================================
# PHOTO PAGE (FIXED)
# =====================================================
if st.session_state.page == "photo":
    st.header("Photo Input 📷")

    img_file = st.file_uploader("Upload skin image", type=["jpg", "png"])

    if img_file and st.button("Analyze"):

        st.info("AI is analyzing image...")

        # MODEL PREDICTION
        label, confidence, cam_image = process_image_with_gradcam(img_file)

        # SAFE IMAGE NAME EXTRACTION
        import os
        img_name = os.path.splitext(img_file.name)[0].strip()

        # dataset lookup
        match = df[df['image_id'] == img_name]

        # SHOW ACTUAL LABEL IF EXISTS
        if not match.empty:
            true_label = match['dx'].values[0]
            st.success(f"Actual Label: {true_label}")
        else:
            st.info("ℹ External image (no ground truth available)")

        # PREDICTION OUTPUT
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")
        # 🔥 SMART WARNING SYSTEM
        if confidence > 0.8:
            st.success("High confidence prediction ✅")
        elif confidence > 0.5:
            st.warning("Moderate confidence ⚠️")
        else:
             st.error("Low confidence ❗ Please upload a clearer skin image")
        st.image(cam_image, caption="Grad-CAM Visualization 🔥")

# =====================================================
# BACK BUTTON
# =====================================================
if st.session_state.page != "home":
    if st.button("Back"):
        go_to_page("home")



