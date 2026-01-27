import torch
import cv2
from model import EfficientNetClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EfficientNetClassifier()
model.load_state_dict(torch.load("efficientnet_masked.pth", map_location=device))
model.to(device)
model.eval()

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)) / 255.0

img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(img)
    pred = out.argmax(1).item()

print("Predicted class:", pred)
