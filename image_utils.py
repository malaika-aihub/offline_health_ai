
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import pandas as pd
import os, sys


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD LABELS
# =====================================================
labels = list(np.load(resource_path("labels_focal.npy"), allow_pickle=True))

# =====================================================
# LOAD MODEL
# =====================================================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(labels)
)

model.load_state_dict(torch.load(resource_path
("best_model_focal.pth"),
map_location=device))
model.to(device)
model.eval()

# =====================================================
# TRANSFORM (must match training)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# LOAD METADATA (OPTIONAL - for testing only)
# =====================================================
df = pd.read_csv("data/raw/HAM10000_metadata.csv")

def get_true_label(image_id):
    match = df[df['image_id'] == image_id]
    if not match.empty:
        return match['dx'].values[0]
    return None

# =====================================================
# SIMPLE PREDICTION FUNCTION
# =====================================================
def process_image(img_file):

    image = Image.open(img_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = str(labels[pred])

    return label, confidence

# =====================================================
# GRAD-CAM CLASS
# =====================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        output = self.model(x)

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap

# =====================================================
# GRAD-CAM PREDICTION FUNCTION
# =====================================================
def process_image_with_gradcam(img_file):

    image_pil = Image.open(img_file).convert("RGB")
    image = transform(image_pil).unsqueeze(0).to(device)

    # Forward pass
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    label = str(labels[pred])

    # -------- Grad-CAM --------
    cam = GradCAM(model, model.blocks[-1])
    heatmap = cam.generate(image, class_idx=pred)

    # Convert heatmap to image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = np.array(image_pil.resize((224, 224)))
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return label, confidence, superimposed 