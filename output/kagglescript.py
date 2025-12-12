# =========================================================
# RAF-DB 4-Class Emotion Classification with ViT-Base
# =========================================================

import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd


# 1. SETTINGS
# =========================================================
DATASET_DIR = "/kaggle/input/raf-db-dataset"

train_csv = os.path.join(DATASET_DIR, "train_labels.csv")
test_csv  = os.path.join(DATASET_DIR, "test_labels.csv")

# Corrected paths pointing to DATASET/
train_images_dir = os.path.join(DATASET_DIR, "DATASET/train")
test_images_dir  = os.path.join(DATASET_DIR, "DATASET/test")

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Keep only stable classes: angry=1, happy=4, neutral=5, sad=6
ALLOWED_CLASSES = [1, 4, 5, 6]
CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

print("Using device:", DEVICE)

# =========================================================
# 2. TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05)),
    transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
    transforms.ToTensor(),  # Convert to tensor first
    transforms.Normalize((0.5,)*3, (0.5,)*3),
    transforms.RandomErasing(p=0.2, scale=(0.02,0.15), ratio=(0.3,3))  # After tensor
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# =========================================================
# 3. DATASET CLASS
# =========================================================
class RAFDBDataset(Dataset):
    def __init__(self, images_dir, labels_csv, allowed_classes=None, transform=None):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

        if allowed_classes is not None:
            self.labels_df = self.labels_df[self.labels_df['label'].isin(allowed_classes)]

        # Map old labels to 0..N-1
        self.label_map = {old: i for i, old in enumerate(sorted(allowed_classes))}
        self.samples = list(zip(self.labels_df['image'], self.labels_df['label']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]

        # The dataset is organized in subfolders 1-7
        folder_name = str(label)  # Map CSV label to subfolder
        img_path = os.path.join(self.images_dir, folder_name, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        new_label = self.label_map[label]
        return img, new_label

# =========================================================
# 4. DATALOADERS
# =========================================================
train_dataset = RAFDBDataset(
    images_dir=train_images_dir,
    labels_csv=train_csv,
    allowed_classes=ALLOWED_CLASSES,
    transform=train_transform
)

test_dataset = RAFDBDataset(
    images_dir=test_images_dir,
    labels_csv=test_csv,
    allowed_classes=ALLOWED_CLASSES,
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# =========================================================
# 5. MODEL
# =========================================================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# =========================================================
# 6. TRAINING LOOP
# =========================================================
train_losses = []
train_accs = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item())

    scheduler.step()
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

# =========================================================
# 7. EVALUATION
# =========================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix plot
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("Confusion Matrix")
os.makedirs("output", exist_ok=True)
plt.savefig("output/confusion_matrix.png")
plt.close()

# =========================================================
# 8. SAVE MODEL AND METRICS
# =========================================================
torch.save(model.state_dict(), "output/vit_rafdb_4class.pth")

plt.figure()
plt.plot(train_losses, label="Loss")
plt.legend()
plt.savefig("output/train_loss.png")
plt.close()

plt.figure()
plt.plot(train_accs, label="Accuracy")
plt.legend()
plt.savefig("output/train_acc.png")
plt.close()

with open("output/classification_report.txt", "w") as f:
    f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

print("Training complete. All outputs saved in 'output/'")
