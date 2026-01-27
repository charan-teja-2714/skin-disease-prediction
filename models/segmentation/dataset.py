import os
import cv2
import torch
from torch.utils.data import Dataset

class ISICSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=128):
        self.samples = []
        self.img_size = img_size

        for img in os.listdir(image_dir):
            if not img.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(image_dir, img)
            mask_name = img.replace(".jpg", "_segmentation.png")
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))

        print(f"✅ Valid segmentation samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Corrupted file: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask