import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PATHS
# =====================================================
CSV_PATH = "data/raw/HAM10000_metadata.csv"
IMAGE_DIR = "data/raw"

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

labels = sorted(df['dx'].unique())
label_map = {label: idx for idx, label in enumerate(labels)}
df['label'] = df['dx'].map(label_map)

# =====================================================
# REMOVE MISSING IMAGES
# =====================================================
def image_exists(row):
    img_name = row['image_id'] + ".jpg"
    path1 = os.path.join(IMAGE_DIR, "HAM10000_images_part_1", img_name)
    path2 = os.path.join(IMAGE_DIR, "HAM10000_images_part_2", img_name)
    return os.path.exists(path1) or os.path.exists(path2)

df = df[df.apply(image_exists, axis=1)].reset_index(drop=True)

print("✅ Clean dataset size:", len(df))

# =====================================================
# SPLIT (NO OVERSAMPLING)
# =====================================================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# =====================================================
# TRANSFORMS (AUGMENTATION ALREADY INCLUDED)
# =====================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# DATASET
# =====================================================
class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id'] + ".jpg"

        path1 = os.path.join(self.img_dir, "HAM10000_images_part_1", img_name)
        path2 = os.path.join(self.img_dir, "HAM10000_images_part_2", img_name)

        if os.path.exists(path1):
            image = Image.open(path1).convert("RGB")
        else:
            image = Image.open(path2).convert("RGB")

        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# LOADERS
# =====================================================
train_loader = DataLoader(
    SkinDataset(train_df, IMAGE_DIR, train_transform),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    SkinDataset(val_df, IMAGE_DIR, val_transform),
    batch_size=32,
    shuffle=False
)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# MODEL (KEEP B0)
# =====================================================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=len(labels),
    drop_rate=0.3
).to(device)

# =====================================================
# FOCAL LOSS (ONLY LOSS FUNCTION)
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

criterion = FocalLoss(alpha=1, gamma=2)

# =====================================================
# OPTIMIZER
# =====================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

# =====================================================
# TRAIN SETUP
# =====================================================
EPOCHS = 20

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

# =====================================================
# EARLY STOPPING
# =====================================================
patience = 7
counter = 0
best_acc = 0

# =====================================================
# TRAIN LOOP
# =====================================================
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # VALIDATION
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")

    # SAVE BEST MODEL
    if acc > best_acc:
        best_acc = acc
        counter = 0

        torch.save(model.state_dict(), "best_model_focal.pth")
        np.save("labels_focal.npy", list(label_map.keys()))

        print("🔥 Best model saved!")
    else:
        counter += 1

    # EARLY STOPPING
    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

    scheduler.step(acc)

# =====================================================
# FINAL EVALUATION
# =====================================================
print("\n🔁 Loading best model...")

model.load_state_dict(torch.load("best_model_focal.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels_batch in val_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\n✅ Training Complete!")
print("Best Accuracy:", best_acc)
print(classification_report(all_labels, all_preds))
