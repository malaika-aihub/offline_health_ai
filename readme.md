# 🧠 Offline Health AI  for skin cancer  detection and other skin related problems

An AI-powered offline healthcare assistant designed to help users analyze skin conditions using images, voice, and text — built for low-resource environments with limited internet access.

---

## 🚀 Features

 Image-based Diagnosis (Primary):

* Uses **EfficientNet (deep learning model)** trained on HAM10000 dataset
* Predicts skin disease from uploaded image
* Provides **confidence score**
* Supports **Grad-CAM visualization** (explains model decision)

---

 Voice Input (Optional):

* Converts speech → text using Whisper
* Detects whether input is skin-related

---

 Text Input (Optional):

* User can describe symptoms manually
* Processed using lightweight logic

---

###  AI Safety Assistant (Gemma 4)

* Does NOT diagnose disease
* Provides:

  * Possible cause (safe hint)
  * Safety precautions
  * Basic advice
* Encourages user to upload image for better analysis

---

## 📊 Dataset

This project uses the HAM10000 Skin Lesion Dataset.

Source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

License:
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

License Link:
https://creativecommons.org/licenses/by-nc-sa/4.0/

### 📌 License Terms Summary:

* You are free to share and adapt the dataset
* You must give appropriate credit
* You cannot use it for commercial purposes
* Any modified version must use same license (ShareAlike)

---

## Tech Stack

* Python
* PyTorch
* EfficientNet (via timm)
* OpenCV (Grad-CAM)
* Whisper (speech-to-text)
* Transformers (Gemma 4)
* Streamlit (UI)

---

##  How It Works

1. User uploads skin image
2. Model predicts disease class
3. Confidence score calculated
4. Grad-CAM highlights affected area
5. Gemma provides safety guidance

---

 Disclaimer

* This system is **NOT a medical diagnosis tool**
* It is designed for **educational and assistive purposes only**
* Always consult a qualified medical professional for serious conditions

---

 Hackathon Goal:

To build an **offline-capable AI health assistant** that:

* Works in low-resource environments
* Reduces dependency on internet
* Provides early awareness for skin conditions

---

Developed as part of the **Gemma 4 Good Hackathon**
Focused on real-world healthcare accessibility using AI.

---
