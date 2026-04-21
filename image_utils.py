
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import os, sys
import cv2
#------------------------
# RESOURCE PATH (for deployment safety)
# ------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LABELS
#------------------------
labels = list(np.load(resource_path("labels_focal.npy"), allow_pickle=True))

# ------------------------
# MODEL LOAD
# ------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(labels)
)

model.load_state_dict(torch.load(resource_path("best_model_focal.pth"),
                                 map_location=device))

model.to(device)
model.eval()

#------------------------
# TRANSFORM
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ------------------------
# 🧠 DISEASE KNOWLEDGE BASE (RULE LAYER)
# ------------------------
SKIN_KNOWLEDGE = {
    "nv": {
        "name": "Melanocytic Nevi (Normal Mole)",
        "type": "🟢 Benign",
        "symptoms": ["Small mole", "Stable size", "No pain", "No bleeding"],
        "advice": "Usually harmless. Monitor for changes."
    },

    "mel": {
        "name": "Melanoma",
        "type": "🔴 Skin Cancer (Dangerous)",
        "symptoms": ["Irregular mole", "Changing color", "Asymmetry", "Dark patch"],
        "advice": "🚨 URGENT: Immediate dermatologist consultation required."
    },

    "bcc": {
        "name": "Basal Cell Carcinoma",
        "type": "🔴 Skin Cancer",
        "symptoms": ["Pearly bump", "Non-healing sore", "Bleeding lesion"],
        "advice": "Medical treatment required. Slow but destructive."
    },

    "akiec": {
        "name": "Actinic Keratosis",
        "type": "🟠 Pre-cancer",
        "symptoms": ["Rough scaly patch", "Sun damage", "Dry crusted lesion"],
        "advice": "⚠️ Can become cancer. Dermatologist check needed."
    },

    "bkl": {
        "name": "Benign Keratosis",
        "type": "🟢 Benign",
        "symptoms": ["Warty lesion", "Brown patch", "Rough skin"],
        "advice": "Harmless condition. Monitor changes."
    },

    "df": {
        "name": "Dermatofibroma",
        "type": "🟢 Benign",
        "symptoms": ["Hard bump", "Brown spot", "Firm nodule"],
        "advice": "No treatment needed unless painful."
    },

    "vasc": {
        "name": "Vascular Lesion",
        "type": "🟢 Usually Benign",
        "symptoms": ["Red/blue spots", "Blood vessel marks"],
        "advice": "Usually harmless. Check if growing."
    }
}


# ------------------------
# GET DISEASE INFO
# ------------------------
def get_disease_info(label):
    return SKIN_KNOWLEDGE.get(label, {
        "name": "Unknown Disease",
        "type": "⚠️ Not Found",
        "symptoms": ["No data available"],
        "advice": "Consult dermatologist."
    })



def simple_gradcam(image_pil):
    image = np.array(image_pil.resize((224, 224)))

    # dummy heatmap example (replace with your real CAM logic if already exists)
    heatmap = np.random.rand(224, 224)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

def process_image_with_gradcam(img_file):

    image_pil = Image.open(img_file).convert("RGB")
    image = transform(image_pil).unsqueeze(0).to(device)

    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    label = str(labels[pred])

    cam_image = simple_gradcam(image_pil)

    disease_info = get_disease_info(label)

    return {
        "label": label,
        "confidence": confidence,
        "image": cam_image,
        "disease_info": disease_info
    }