import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import ISICClassificationDataset
from model import EfficientNetClassifier

IMAGE_DIR = "../../data/raw/images"
CSV_PATH = "../../data/raw/metadata.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# IMPROVED DATA AUGMENTATION
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Load full dataset
full_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH)

# TRAIN/VAL SPLIT (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"📊 Train: {train_size}, Val: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Model with DROPOUT for regularization
model = EfficientNetClassifier(num_classes=7).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Added weight decay

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

epochs = 10
best_val_acc = 0.0

print("\n🚀 Training with validation...")

for epoch in range(epochs):
    # TRAINING
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    
    train_acc = train_correct / train_total
    
    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(1)
            
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    
    print(f"✅ Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "efficientnet_best.pth")
        print(f"   💾 Saved best model (Val Acc: {val_acc:.4f})")
    
    # Adjust learning rate
    scheduler.step(val_acc)

print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.4f}")
print("📁 Best model saved as: efficientnet_best.pth")


import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Import your custom dataset and the new model
from dataset import ISICClassificationDataset
from model_updated import EfficientNetV2LFeatureExtractor

# --- CONFIGURATION ---
IMAGE_DIR = "../../data/raw/images"
CSV_PATH = "../../data/raw/metadata.csv"
BATCH_SIZE = 32 # Can be higher since we are only inferencing, not backpropagating
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# --- TRANSFORMS ---
# Note: For feature extraction, we generally want minimal augmentation (Resize + Normalize)
# LightGBM handles the variance. Heavy augmentation is less critical here than in end-to-end CNNs.
transform = A.Compose([
    A.Resize(224, 224), # EfficientNetV2 ideal size usually starts at 224 or 260
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- DATA LOADING ---
print("📂 Loading Dataset...")
# Assuming ISICClassificationDataset takes a 'transform' arg
full_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH, transform=transform)

# Split (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"📊 Train samples: {train_size}, Validation samples: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- STEP 1: FEATURE EXTRACTION ---
print("\n🚀 Initializing EfficientNetV2L Feature Extractor...")
model = EfficientNetV2LFeatureExtractor().to(device)
model.eval() # Set to evaluation mode (disables dropout/batchnorm updates)

def extract_features(loader, desc="Extracting"):
    """Passes images through the CNN to get feature vectors."""
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            
            # Get (Batch_Size, 1280) vectors
            features = model(images)
            
            # Move to CPU and store
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Stack into one massive NumPy array
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y

print("\n📸 extracting Training Features (this takes time)...")
X_train, y_train = extract_features(train_loader, desc="[Train] Features")

print("\n📸 extracting Validation Features...")
X_val, y_val = extract_features(val_loader, desc="[Val] Features")

print(f"✅ Feature Extraction Complete.")
print(f"   Train shape: {X_train.shape}")
print(f"   Val shape:   {X_val.shape}")

# --- STEP 2: TRAIN LIGHTGBM ---
print("\n🌲 Training LightGBM Classifier...")

# Parameters optimized for ISIC imbalance and high accuracy
lgbm = LGBMClassifier(
    n_estimators=1200,      # Number of boosting rounds
    learning_rate=0.05,     # Slower learning for better generalization
    num_leaves=31,          # Control tree complexity
    boosting_type='gbdt',
    objective='multiclass',
    num_class=7,            # ISIC 2018 has 7 classes
    is_unbalance=True,      # CRITICAL: Auto-weights minority classes (Melanoma)
    metric='multi_logloss',
    # n_jobs=-1,              # Use all CPU cores
    verbose=-1
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    # eval_metric='multi_logloss', # Removed to avoid warning in newer versions
    # early_stopping_rounds=50     # Optional: Stop if val score stops improving
)

# --- STEP 3: EVALUATION ---
print("\n🏆 Evaluating Model...")
y_pred = lgbm.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)

print(f"\n🎉 Final Validation Accuracy: {val_acc:.4f}")
print("-" * 60)
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("-" * 60)

# Save the LightGBM model (it's very small)
import joblib
joblib.dump(lgbm, "lgbm_skin_cancer_model.pkl")
print("💾 Saved LightGBM model to 'lgbm_skin_cancer_model.pkl'")


# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import joblib
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib


# # Import your custom dataset and the new model
# from dataset import ISICClassificationDataset
# from model_updated import EfficientNetV2LFeatureExtractor

# # --- CONFIGURATION ---
# IMAGE_DIR = "../../data/raw/images"
# CSV_PATH = "../../data/raw/metadata.csv"
# BATCH_SIZE = 16  # Reduced batch size for fine-tuning stability
# device = "cuda" if torch.cuda.is_available() else "cpu"

# def extract_features(loader, model, desc="Extracting"):
#     """Passes images through the CNN to get feature vectors."""
#     model.eval()
#     features_list = []
#     labels_list = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(loader, desc=desc):
#             images = images.to(device)
#             # Get (Batch_Size, 1280) vectors
#             features = model(images)
#             features_list.append(features.cpu().numpy())
#             labels_list.append(labels.numpy())
            
#     X = np.vstack(features_list)
#     y = np.concatenate(labels_list)
#     return X, y

# # --- MANDATORY WINDOWS MAIN GUARD ---
# if __name__ == '__main__':
#     print(f"🔥 Using device: {device}")

#     # 1. TRANSFORMS (Added subtle augmentation for the fine-tuning phase)
#     train_transform = A.Compose([
#         A.Resize(224, 224),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])

#     val_transform = A.Compose([
#         A.Resize(224, 224),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])

#     # 2. DATA LOADING
#     print("📂 Loading Dataset...")
#     full_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH) # Logic for transforms inside loop below

#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
#     # Manually assign transforms to splits
#     train_ds.dataset.transform = train_transform
#     val_ds.dataset.transform = val_transform

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     # 3. INITIALIZE MODEL
#     print("\n🚀 Initializing EfficientNetV2L...")
#     model = EfficientNetV2LFeatureExtractor().to(device)

#     # 4. PHASE 1: FINE-TUNING (The "Warm-up")
#     # This teaches the ImageNet weights how to look at skin lesions
#     print("\n🔥 Phase 1: Fine-tuning backbone for 3 epochs...")
#     fine_tune_head = nn.Linear(1280, 7).to(device) # Temporary head
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(list(model.parameters()) + list(fine_tune_head.parameters()), lr=1e-4)

#     for epoch in range(3):
#         model.train()
#         fine_tune_head.train()
#         for images, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/3"):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             features = model(images)
#             outputs = fine_tune_head(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#     # 5. PHASE 2: FEATURE EXTRACTION
#     print("\n📸 Extracting Skin-Aware Features...")
#     X_train, y_train = extract_features(train_loader, model, desc="[Train] Features")
#     X_val, y_val = extract_features(val_loader, model, desc="[Val] Features")

#     # 6. PHASE 3: TRAIN LIGHTGBM
#     print("\n🌲 Training LightGBM Classifier...")
#     lgbm = LGBMClassifier(
#         n_estimators=2000,
#         learning_rate=0.03,
#         num_leaves=63,        # Increased for deeper feature complexity
#         boosting_type='gbdt',
#         objective='multiclass',
#         num_class=7,
#         is_unbalance=True,    # Fixes the class 5 & 6 (0.00 accuracy) issue
#         extra_trees=True,     # Better for CNN feature vectors
#         importance_type='gain',
#         n_jobs=-1,
#         verbose=-1
#     )

#     lgbm.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#     )

#     # 7. EVALUATION
#     print("\n🏆 Evaluating Final Hybrid Model...")
#     y_pred = lgbm.predict(X_val)
#     val_acc = accuracy_score(y_val, y_pred)

#     print(f"\n🎉 Final Validation Accuracy: {val_acc:.4f}")
#     print("-" * 60)
#     print("Classification Report:")
#     print(classification_report(y_val, y_pred))
#     print("-" * 60)

#     # SAVE
#     joblib.dump(lgbm, "lgbm_skin_cancer_model.pkl")
#     torch.save(model.state_dict(), "tuned_efficientnet_backbone.pth")
#     print("💾 Saved both Hybrid Classifier and Tuned Backbone.")