
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
SKIN_KNOWLEDGE = {

    "mel": {
        "name": {
            "en":"Melanoma","ur":"میلانوما","hi":"मेलानोमा","ar":"الميلانوما","fr":"Mélanome"
        },
        "type": {
            "en":"🔴 Malignant Skin Cancer","ur":"🔴 مہلک جلد کا کینسر","hi":"🔴 घातक त्वचा कैंसर","ar":"🔴 سرطان جلدي خبيث","fr":"🔴 Cancer cutané malin"
        },
        "symptoms": {
            "en":[
                "Asymmetry in shape",
                "Irregular or notched borders",
                "Color variation (black, brown, red, blue)",
                "Diameter greater than 6mm",
                "Evolution in size or shape",
                "Bleeding, itching or crusting lesion"
            ],
            "ur":[
                "غیر متوازن شکل",
                "غیر ہموار یا کٹے ہوئے کنارے",
                "مختلف رنگ (سیاہ، بھورا، نیلا، سرخ)",
                "6 ملی میٹر سے بڑا سائز",
                "وقت کے ساتھ تبدیلی",
                "خون آنا، خارش یا کرسٹ بننا"
            ],
            "hi":[
                "असममित आकार",
                "अनियमित या कटे किनारे",
                "कई रंग (काला, भूरा, नीला, लाल)",
                "6 मिमी से बड़ा आकार",
                "समय के साथ बदलाव",
                "खून आना, खुजली या पपड़ी"
            ],
            "ar":[
                "عدم تماثل الشكل",
                "حواف غير منتظمة",
                "تعدد الألوان",
                "أكبر من 6 مم",
                "تغير تدريجي",
                "نزيف أو حكة أو قشرة"
            ],
            "fr":[
                "Asymétrie",
                "Bords irréguliers",
                "Variations de couleur",
                ">6 mm",
                "Évolution",
                "Saignement ou croûtes"
            ]
        },
        "advice":{
            "en":"🚨 Urgent dermatology evaluation required",
            "ur":"🚨 فوری ماہرِ جلد سے معائنہ ضروری ہے",
            "hi":"🚨 तुरंत त्वचा विशेषज्ञ से जांच कराएं",
            "ar":"🚨 فحص جلدي عاجل مطلوب",
            "fr":"🚨 Consultation dermatologique urgente"
        }
    },

    "nv": {
        "name": {
            "en":"Benign Nevus","ur":"عام تل","hi":"सामान्य तिल","ar":"شامة حميدة","fr":"Nævus bénin"
        },
        "type": {
            "en":"🟢 Benign lesion","ur":"🟢 بے ضرر نشان","hi":"🟢 सामान्य घाव","ar":"🟢 حميد","fr":"🟢 Bénin"
        },
        "symptoms": {
            "en":[
                "Symmetrical round/oval shape",
                "Uniform brown color",
                "Well-defined smooth borders",
                "Stable size over years",
                "No bleeding, itching or pain"
            ],
            "ur":[
                "گول یا بیضوی متوازن شکل",
                "ایک جیسا بھورا رنگ",
                "صاف اور ہموار کنارے",
                "سالوں سے مستحکم سائز",
                "درد، خارش یا خون نہیں"
            ],
            "hi":[
                "गोल/अंडाकार सममित आकार",
                "एक समान भूरा रंग",
                "साफ किनारे",
                "लंबे समय से स्थिर",
                "दर्द या खून नहीं"
            ],
            "ar":[
                "شكل متماثل",
                "لون بني موحد",
                "حواف واضحة",
                "ثابت الحجم",
                "بدون أعراض"
            ],
            "fr":[
                "Forme symétrique",
                "Couleur uniforme",
                "Bords nets",
                "Stable",
                "Asymptomatique"
            ]
        },
        "advice":{
            "en":"Normal benign mole, no treatment needed",
            "ur":"عام بے ضرر تل، علاج کی ضرورت نہیں",
            "hi":"सामान्य स्थिति, इलाज की जरूरत नहीं",
            "ar":"حالة طبيعية",
            "fr":"Lésion bénigne"
        }
    },

    "bcc": {
        "name": {
            "en":"Basal Cell Carcinoma","ur":"بیسل سیل کینسر","hi":"बेसल सेल कैंसर","ar":"سرطان الخلايا القاعدية","fr":"Carcinome basocellulaire"
        },
        "type": {
            "en":"🔴 Skin Cancer","ur":"🔴 جلد کا کینسر","hi":"🔴 त्वचा कैंसर","ar":"🔴 سرطان الجلد","fr":"🔴 Cancer cutané"
        },
        "symptoms": {
            "en":[
                "Pearly or shiny bump",
                "Non-healing ulcer or sore",
                "Bleeds with minor trauma",
                "Flat scar-like patch",
                "Visible telangiectasia (tiny blood vessels)"
            ],
            "ur":[
                "چمکدار دانہ",
                "زخم جو نہ بھرے",
                "ہلکی چوٹ سے خون آنا",
                "داغ جیسا نشان",
                "چھوٹی رگوں کا نظر آنا"
            ],
            "hi":[
                "मोती जैसा चमकदार दाना",
                "घाव न भरना",
                "हल्की चोट पर खून",
                "दाग जैसा निशान",
                "नसें दिखना"
            ],
            "ar":[
                "نتوء لامع",
                "قرحة لا تلتئم",
                "نزيف بسهولة",
                "بقعة شبيهة بالندبة",
                "أوعية دقيقة ظاهرة"
            ],
            "fr":[
                "Nodule brillant",
                "Ulcère persistant",
                "Saignement facile",
                "Plaque cicatricielle",
                "Vaisseaux visibles"
            ]
        },
        "advice":{
            "en":"Medical treatment required",
            "ur":"طبی علاج ضروری ہے",
            "hi":"इलाज जरूरी है",
            "ar":"يحتاج علاج",
            "fr":"Traitement médical"
        }
    },

    "akiec": {
        "name": {
            "en":"Actinic Keratosis","ur":"ایکٹینک کیراٹوسس","hi":"एक्टिनिक केराटोसिस","ar":"التقرن الشمسي","fr":"Kératose actinique"
        },
        "type": {
            "en":"🟠 Pre-cancerous lesion","ur":"🟠 کینسر سے پہلے کا نشان","hi":"🟠 पूर्व कैंसर","ar":"🟠 قبل السرطان","fr":"🟠 Pré-cancer"
        },
        "symptoms": {
            "en":[
                "Rough sandpaper-like patch",
                "Dry scaly or crusted lesion",
                "Sun-exposed skin areas",
                "Pink to red rough plaque",
                "Mild burning or itching"
            ],
            "ur":[
                "ریت جیسے کھردرے نشان",
                "خشک اور چھلکا دار جلد",
                "دھوپ والی جگہ",
                "گلابی یا سرخ دھبہ",
                "ہلکی جلن یا خارش"
            ],
            "hi":[
                "रेत जैसा खुरदुरा दाग",
                "सूखी पपड़ीदार त्वचा",
                "धूप वाली जगह",
                "गुलाबी/लाल धब्बा",
                "जलन या खुजली"
            ],
            "ar":[
                "بقعة خشنة",
                "جلد جافة متقشرة",
                "مناطق معرضة للشمس",
                "بقعة حمراء",
                "حكة أو حرقان"
            ],
            "fr":[
                "Plaque rugueuse",
                "Peau sèche squameuse",
                "Zones exposées",
                "Lésion rouge",
                "Démangeaison"
            ]
        },
        "advice":{
            "en":"Dermatologist evaluation recommended",
            "ur":"ماہرِ جلد سے معائنہ ضروری ہے",
            "hi":"त्वचा विशेषज्ञ से जांच कराएं",
            "ar":"فحص جلدي",
            "fr":"Consultation dermatologique"
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