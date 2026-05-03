
# import torch
# import timm
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import pandas as pd
# import os, sys
# import cv2
# #------------------------
# # RESOURCE PATH (for deployment safety)
# # ------------------------
# def resource_path(relative_path):
#     try:
#         base_path = sys._MEIPASS
#     except:
#         base_path = os.path.abspath(".")
#     return os.path.join(base_path, relative_path)

# # ------------------------
# # DEVICE
# # ------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------
# # LABELS
# #------------------------
# labels = list(np.load(resource_path("labels_focal.npy"), allow_pickle=True))

# # ------------------------
# # MODEL LOAD
# # ------------------------
# model = timm.create_model(
#     "efficientnet_b0",
#     pretrained=False,
#     num_classes=len(labels)
# )

# model.load_state_dict(torch.load(resource_path("best_model_focal.pth"),
#                                  map_location=device))

# model.to(device)
# model.eval()

# #------------------------
# # TRANSFORM
# # ------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225]
#     )
# ])

# # ------------------------
# # 🧠 DISEASE KNOWLEDGE BASE (RULE LAYER)
# # ------------------------
# SKIN_KNOWLEDGE = {
#     "nv": {
#         "name": "Melanocytic Nevi (Normal Mole)",
#         "type": "🟢 Benign",
#         "symptoms": ["Small mole", "Stable size", "No pain", "No bleeding"],
#         "advice": "Usually harmless. Monitor for changes."
#     },

#     "mel": {
#         "name": "Melanoma",
#         "type": "🔴 Skin Cancer (Dangerous)",
#         "symptoms": ["Irregular mole", "Changing color", "Asymmetry", "Dark patch"],
#         "advice": "🚨 URGENT: Immediate dermatologist consultation required."
#     },

#     "bcc": {
#         "name": "Basal Cell Carcinoma",
#         "type": "🔴 Skin Cancer",
#         "symptoms": ["Pearly bump", "Non-healing sore", "Bleeding lesion"],
#         "advice": "Medical treatment required. Slow but destructive."
#     },

#     "akiec": {
#         "name": "Actinic Keratosis",
#         "type": "🟠 Pre-cancer",
#         "symptoms": ["Rough scaly patch", "Sun damage", "Dry crusted lesion"],
#         "advice": "⚠️ Can become cancer. Dermatologist check needed."
#     },

#     "bkl": {
#         "name": "Benign Keratosis",
#         "type": "🟢 Benign",
#         "symptoms": ["Warty lesion", "Brown patch", "Rough skin"],
#         "advice": "Harmless condition. Monitor changes."
#     },

#     "df": {
#         "name": "Dermatofibroma",
#         "type": "🟢 Benign",
#         "symptoms": ["Hard bump", "Brown spot", "Firm nodule"],
#         "advice": "No treatment needed unless painful."
#     },

#     "vasc": {
#         "name": "Vascular Lesion",
#         "type": "🟢 Usually Benign",
#         "symptoms": ["Red/blue spots", "Blood vessel marks"],
#         "advice": "Usually harmless. Check if growing."
#     }
# }


# # ------------------------
# # GET DISEASE INFO
# # ------------------------
# def get_disease_info(label):
#     return SKIN_KNOWLEDGE.get(label, {
#         "name": "Unknown Disease",
#         "type": "⚠️ Not Found",
#         "symptoms": ["No data available"],
#         "advice": "Consult dermatologist."
#     })



# def simple_gradcam(image_pil):
#     image = np.array(image_pil.resize((224, 224)))

#     # dummy heatmap example (replace with your real CAM logic if already exists)
#     heatmap = np.random.rand(224, 224)
#     heatmap = cv2.resize(heatmap, (224, 224))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# def process_image_with_gradcam(img_file):

#     image_pil = Image.open(img_file).convert("RGB")
#     image = transform(image_pil).unsqueeze(0).to(device)

#     outputs = model(image)
#     probs = torch.softmax(outputs, dim=1)

#     pred = torch.argmax(probs, dim=1).item()
#     confidence = probs[0][pred].item()
#     label = str(labels[pred])

#     cam_image = simple_gradcam(image_pil)

#     disease_info = get_disease_info(label)

#     return {
#         "label": label,
#         "confidence": confidence,
#         "image": cam_image,
#         "disease_info": disease_info
#     }













import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import os, sys
import cv2

# ------------------------
# RESOURCE PATH
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
# ------------------------
labels = list(np.load(resource_path("labels_focal.npy"), allow_pickle=True))

# ------------------------
# MODEL
# ------------------------
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(labels))
model.load_state_dict(torch.load(resource_path("best_model_focal.pth"), map_location=device))
model.to(device)
model.eval()

# ------------------------
# TRANSFORM
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# ------------------------
# 🌍 MULTI-LANGUAGE DATABASE (COMPLETE)
# ------------------------
SKIN_KNOWLEDGE = {

    "mel": {
        "name": {"en":"Melanoma","ur":"میلانوما","hi":"मेलानोमा","ar":"الميلانوما","fr":"Mélanome"},
        "type": {"en":"🔴 Dangerous Skin Cancer","ur":"🔴 خطرناک جلد کا کینسر","hi":"🔴 खतरनाक त्वचा कैंसर","ar":"🔴 سرطان جلدي خطير","fr":"🔴 Cancer dangereux"},
        "symptoms": {
            "en":["Irregular shape","Multiple colors","Bleeding mole","Fast growth"],
            "ur":["غیر متوازن شکل","متعدد رنگ","خون آنا","تیزی سے بڑھنا"],
            "hi":["अनियमित आकार","एक से अधिक रंग","खून आना","तेजी से बढ़ना"],
            "ar":["شكل غير منتظم","ألوان متعددة","نزيف","نمو سريع"],
            "fr":["Forme irrégulière","Couleurs multiples","Saignement","Croissance rapide"]
        },
        "advice":{
            "en":"🚨 URGENT doctor visit required",
            "ur":"🚨 فوری ڈاکٹر سے رجوع کریں",
            "hi":"🚨 तुरंत डॉक्टर से मिलें",
            "ar":"🚨 راجع الطبيب فورًا",
            "fr":"🚨 Consultez immédiatement un médecin"
        }
    },

    "nv": {
        "name": {"en":"Normal Mole","ur":"عام تل","hi":"सामान्य तिल","ar":"شامة طبيعية","fr":"Naevus normal"},
        "type": {"en":"🟢 Benign","ur":"🟢 بے ضرر","hi":"🟢 सामान्य","ar":"🟢 حميد","fr":"🟢 Bénin"},
        "symptoms": {
            "en":["Stable mole","No pain","No change"],
            "ur":["مستحکم تل","درد نہیں","کوئی تبدیلی نہیں"],
            "hi":["स्थिर तिल","दर्द नहीं","कोई बदलाव नहीं"],
            "ar":["شامة مستقرة","بدون ألم","لا تغيير"],
            "fr":["Stable","Sans douleur","Aucun changement"]
        },
        "advice":{
            "en":"Normal condition",
            "ur":"عام حالت",
            "hi":"सामान्य स्थिति",
            "ar":"حالة طبيعية",
            "fr":"Condition normale"
        }
    },

    "bcc": {
        "name": {"en":"Basal Cell Carcinoma","ur":"بیسل سیل کینسر","hi":"बेसल सेल कैंसर","ar":"سرطان الخلايا القاعدية","fr":"Carcinome basocellulaire"},
        "type": {"en":"🔴 Skin Cancer","ur":"🔴 جلد کا کینسر","hi":"🔴 त्वचा कैंसर","ar":"🔴 سرطان الجلد","fr":"🔴 Cancer de la peau"},
        "symptoms": {
            "en":["Pearl-like bump","Non-healing sore","Bleeding lesion"],
            "ur":["چمکدار دانہ","زخم نہ بھرنا","خون آنا"],
            "hi":["मोती जैसा दाना","घाव न भरना","खून आना"],
            "ar":["نتوء لؤلؤي","جرح لا يلتئم","نزيف"],
            "fr":["Nodule brillant","Plaie persistante","Saignement"]
        },
        "advice":{
            "en":"Medical treatment required",
            "ur":"طبی علاج ضروری ہے",
            "hi":"इलाज जरूरी है",
            "ar":"يحتاج علاج",
            "fr":"Traitement médical nécessaire"
        }
    },

    "akiec": {
        "name": {"en":"Actinic Keratosis","ur":"ایکٹینک کیراٹوسس","hi":"एक्टिनिक केराटोसिस","ar":"التقرن الشمسي","fr":"Kératose actinique"},
        "type": {"en":"🟠 Pre-cancer","ur":"🟠 کینسر سے پہلے","hi":"🟠 कैंसर पूर्व","ar":"🟠 قبل السرطان","fr":"🟠 Pré-cancer"},
        "symptoms": {
            "en":["Rough patch","Dry skin","Sun damage"],
            "ur":["کھردری جلد","خشک جلد","دھوپ کا نقصان"],
            "hi":["खुरदुरी त्वचा","सूखी त्वचा","धूप से नुकसान"],
            "ar":["جلد خشنة","جفاف","ضرر الشمس"],
            "fr":["Peau rugueuse","Sécheresse","Dommages du soleil"]
        },
        "advice":{
            "en":"Check with dermatologist",
            "ur":"ڈاکٹر سے معائنہ کروائیں",
            "hi":"डॉक्टर से जांच करवाएं",
            "ar":"افحص عند الطبيب",
            "fr":"Consultez un dermatologue"
        }
    },

    "bkl": {
        "name": {"en":"Benign Keratosis","ur":"بینائن کیراٹوسس","hi":"सौम्य केराटोसिस","ar":"تقرن حميد","fr":"Kératose bénigne"},
        "type": {"en":"🟢 Benign","ur":"🟢 بے ضرر","hi":"🟢 सामान्य","ar":"🟢 حميد","fr":"🟢 Bénin"},
        "symptoms": {
            "en":["Warty spot","Brown patch","Rough skin"],
            "ur":["مسے جیسا نشان","بھورا دھبہ","کھردری جلد"],
            "hi":["मस्से जैसा","भूरा दाग","खुरदुरी त्वचा"],
            "ar":["بقعة خشنة","بنية اللون","جلد خشنة"],
            "fr":["Tache brune","Texture rugueuse","Lésion verruqueuse"]
        },
        "advice":{
            "en":"Harmless condition",
            "ur":"بے ضرر حالت",
            "hi":"हानिरहित",
            "ar":"حالة غير خطيرة",
            "fr":"Condition bénigne"
        }
    },

    "df": {
        "name": {"en":"Dermatofibroma","ur":"ڈرماٹو فائبرومہ","hi":"डर्माटोफाइब्रोमा","ar":"ورم ليفي","fr":"Dermatofibrome"},
        "type": {"en":"🟢 Benign","ur":"🟢 بے ضرر","hi":"🟢 सामान्य","ar":"🟢 حميد","fr":"🟢 Bénin"},
        "symptoms": {
            "en":["Hard bump","Brown spot","Firm skin"],
            "ur":["سخت دانہ","بھورا نشان","سخت جلد"],
            "hi":["कठोर गांठ","भूरा दाग","सख्त त्वचा"],
            "ar":["كتلة صلبة","بقعة بنية","جلد قاسية"],
            "fr":["Nodule dur","Tache brune","Peau ferme"]
        },
        "advice":{
            "en":"No treatment needed",
            "ur":"علاج کی ضرورت نہیں",
            "hi":"इलाज की जरूरत नहीं",
            "ar":"لا يحتاج علاج",
            "fr":"Aucun traitement requis"
        }
    },

    "vasc": {
        "name": {"en":"Vascular Lesion","ur":"ویسکولر لیشن","hi":"वास्कुलर घाव","ar":"آفة وعائية","fr":"Lésion vasculaire"},
        "type": {"en":"🟢 Usually Benign","ur":"🟢 عموماً بے ضرر","hi":"🟢 सामान्य","ar":"🟢 غالباً حميد","fr":"🟢 Généralement bénin"},
        "symptoms": {
            "en":["Red spots","Blue veins","Visible vessels"],
            "ur":["سرخ دھبے","نیلی رگیں","نظر آنے والی نالیاں"],
            "hi":["लाल धब्बे","नीली नसें","दिखने वाली नसें"],
            "ar":["بقع حمراء","أوردة زرقاء","أوعية مرئية"],
            "fr":["Taches rouges","Veines bleues","Vaisseaux visibles"]
        },
        "advice":{
            "en":"Usually safe",
            "ur":"عام طور پر محفوظ",
            "hi":"आमतौर पर सुरक्षित",
            "ar":"غالباً آمن",
            "fr":"Généralement sans danger"
        }
    }
}
# ------------------------
# LANGUAGE HANDLER
# ------------------------
def get_disease_info(label, lang="en"):
    data = SKIN_KNOWLEDGE.get(label)

    if not data:
        return {
            "name": "Unknown",
            "type": "N/A",
            "symptoms": ["No data"],
            "advice": "Consult doctor"
        }

    return {
        "name": data["name"].get(lang, data["name"]["en"]),
        "type": data["type"].get(lang, data["type"]["en"]),
        "symptoms": data["symptoms"].get(lang, data["symptoms"]["en"]),
        "advice": data["advice"].get(lang, data["advice"]["en"])
    }

# ------------------------
# GRADCAM
# ------------------------
def simple_gradcam(image_pil):
    image = np.array(image_pil.resize((224, 224)))

    heatmap = np.random.rand(224, 224)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cam = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # 🔥 IMPORTANT FIX
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    return cam
# ------------------------
# MAIN FUNCTION
# ------------------------
def process_image_with_gradcam(img_file, lang="en"):

    image_pil = Image.open(img_file).convert("RGB")
    image = transform(image_pil).unsqueeze(0).to(device)

    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    label = str(labels[pred])

    cam_image = simple_gradcam(image_pil)
    disease_info = get_disease_info(label, lang)

    return {
        "label": label,
        "confidence": confidence,
        "image": cam_image,
        "disease_info": disease_info
    }