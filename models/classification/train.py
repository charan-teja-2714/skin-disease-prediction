import torch
from torch.utils.data import DataLoader
from dataset import ISICClassificationDataset
from model import EfficientNetClassifier
from tqdm import tqdm

IMAGE_DIR = "../../data/raw/images"
MASK_DIR = "../../data/raw/masks"
CSV_PATH = "../../data/raw/metadata.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Using device:", device)

dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = EfficientNetClassifier().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

for epoch in range(epochs):
    model.train()
    correct = total = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct/total)

    print(f"✅ Epoch {epoch+1} Accuracy: {correct/total:.4f}")

torch.save(model.state_dict(), "efficientnet_best.pth")
print("🎉 Classification model saved")
