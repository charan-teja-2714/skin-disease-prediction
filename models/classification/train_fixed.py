import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import ISICClassificationDataset
from model import EfficientNetClassifier

IMAGE_DIR = "../../data/raw/images"
CSV_PATH = "../../data/raw/metadata.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# Load dataset
full_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH)

# TRAIN/VAL SPLIT (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"📊 Train: {train_size}, Val: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Model
model = EfficientNetClassifier(num_classes=7).to(device)

# Loss and optimizer with regularization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

epochs = 10
best_val_acc = 0.0

print("\n🚀 Training with validation...")

for epoch in range(epochs):
    # TRAINING
    model.train()
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    
    scheduler.step(val_acc)

print(f"\n🎉 Training complete! Best Val Acc: {best_val_acc:.4f}")
print("📁 Best model saved as: efficientnet_best.pth")
