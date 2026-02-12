# Skin Disease Prediction System - Setup & Execution Guide

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test.py
```

### 2. Data Preparation
```bash
# Create data directory structure
mkdir -p data/raw/images
mkdir -p data/raw/masks
mkdir -p data/processed

# Place your ISIC dataset in:
# - data/raw/images/ (skin lesion images)
# - data/raw/masks/ (segmentation masks)
# - data/raw/metadata.csv (classification labels)
```

## 🎯 Training Pipeline

### Step 1: Train Segmentation Model (U-Net)
```bash
cd models/segmentation
python train.py
```
**Expected Output:**
- Training progress with Dice loss
- Model saved as `unet_isic_gpu_safe.pth`
- Training time: ~30-60 minutes (depending on dataset size)

### Step 2: Train Classification Model (EfficientNet)
```bash
cd models/classification
python train.py
```
**Expected Output:**
- Training progress with accuracy metrics
- Model saved as `efficientnet_best.pth`
- Training time: ~45-90 minutes

### Step 3: Evaluate Models
```bash
# Classification evaluation
cd models/classification
python infer.py

# Fairness evaluation
cd models/fairness
python evaluate.py
```

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Segmentation Metrics
- **Dice Coefficient**: 2 * |A ∩ B| / (|A| + |B|)
- **IoU (Jaccard Index)**: |A ∩ B| / |A ∪ B|
- **Pixel Accuracy**: Correctly classified pixels / Total pixels

### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Calibration**: Prediction confidence matches actual accuracy

## 🔬 Model Evaluation Scripts

### Create Comprehensive Evaluation Script
```python
# models/evaluation/evaluate_all.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy, precision, recall, f1

def dice_coefficient(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def evaluate_segmentation(model, test_loader, device):
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds) > 0.5
            
            dice = dice_coefficient(preds, masks)
            dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice:.4f}")
    
    return avg_dice
```

## 🌐 Running the Application

### Backend Setup
```bash
cd backend
# Create main FastAPI app
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
# If using React/Next.js
npm install
npm run dev

# If using Streamlit
streamlit run app.py
```

## 📈 Expected Training Outputs

### Segmentation Training Output:
```
🔥 Using device: cuda
Epoch 1/2: 100%|██████████| 1000/1000 [15:30<00:00, loss=0.2341]
✅ Epoch 1 Avg Loss: 0.2341
Epoch 2/2: 100%|██████████| 1000/1000 [15:28<00:00, loss=0.1892]
✅ Epoch 2 Avg Loss: 0.1892
🎉 Model saved as unet_isic_gpu_safe.pth
```

### Classification Training Output:
```
🔥 Using device: cuda
Epoch 1/5: 100%|██████████| 500/500 [12:45<00:00, loss=1.2341, acc=0.7234]
✅ Epoch 1 Accuracy: 0.7234
Epoch 2/5: 100%|██████████| 500/500 [12:43<00:00, loss=0.8921, acc=0.8156]
✅ Epoch 2 Accuracy: 0.8156
...
🎉 Classification model saved
```

## 🔧 Troubleshooting

### Common Issues:
1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Missing Data**: Ensure data is in correct directory structure
3. **Import Errors**: Check virtual environment activation

### Performance Optimization:
- Use mixed precision training: `torch.cuda.amp`
- Implement data loading optimization
- Use gradient accumulation for larger effective batch sizes

## 📊 Model Performance Benchmarks

### Expected Performance:
- **Segmentation Dice Score**: 0.85-0.92
- **Classification Accuracy**: 0.80-0.88
- **Inference Time**: <100ms per image
- **Model Size**: 
  - U-Net: ~30MB
  - EfficientNet: ~20MB

## 🚀 Deployment

### Local Deployment:
```bash
# Start backend
cd backend && python main.py

# Start frontend
cd frontend && npm start
```

### Docker Deployment:
```bash
docker build -t skin-disease-app .
docker run -p 8000:8000 skin-disease-app
```

## 📝 Next Steps

1. **Model Optimization**: Implement model quantization and pruning
2. **Data Augmentation**: Add more sophisticated augmentation techniques
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Real-time Inference**: Optimize for mobile deployment
5. **Clinical Validation**: Test with medical professionals