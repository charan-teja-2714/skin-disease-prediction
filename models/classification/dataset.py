import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

class ISICClassificationDataset(Dataset):
    def __init__(self, image_dir, csv_path, img_size=224):
        self.image_dir = image_dir
        self.img_size = img_size

        df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in df.iterrows():
            image_id = row["image"]

            # determine class from one-hot encoding
            label = None
            for idx, cls in enumerate(CLASS_NAMES):
                if row[cls] == 1:
                    label = idx
                    break

            if label is None:
                continue

            img_path = os.path.join(image_dir, image_id + ".jpg")
            if os.path.exists(img_path):
                self.samples.append((img_path, label))

        print(f"✅ Classification samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
