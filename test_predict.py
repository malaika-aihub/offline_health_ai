from PIL import Image
from image_utils import process_image, get_true_label
import os

# image path
img_path = "data/raw/HAM10000_images_part_1/ISIC_0026769.jpg"

# 🔍 CHECK YAHAN KARO
print("Exists:", os.path.exists(img_path))
print("Absolute path:", os.path.abspath(img_path))


# prediction
label, confidence = process_image(img_path)

print("PREDICTED:", label)
print("CONFIDENCE:", confidence)

# actual label (ground truth)
true_label = get_true_label("ISIC_0026769")

print("ACTUAL:", true_label)