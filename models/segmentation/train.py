import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ISICSegmentationDataset
from unet import UNet
from utils import dice_loss

IMAGE_DIR = "../../data/raw/images"
MASK_DIR = "../../data/raw/masks"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Using device:", device)

dataset = ISICSegmentationDataset(IMAGE_DIR, MASK_DIR, img_size=128)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 2

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "unet_isic_gpu_safe.pth")
print("🎉 Model saved as unet_isic_gpu_safe.pth")