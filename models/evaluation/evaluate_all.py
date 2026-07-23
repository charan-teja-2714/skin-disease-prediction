import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
import sys

# Add model paths
sys.path.append('../classification')
sys.path.append('../segmentation')

from model import EfficientNetClassifier
from unet import UNet
from dataset import ISICClassificationDataset
from torch.utils.data import DataLoader

# Import segmentation dataset separately (different module)
sys.path.append('../segmentation')
from dataset import ISICSegmentationDataset


class ModelEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}

    def evaluate_classification(self, model_path, test_loader, class_names=None):
        """Comprehensive classification evaluation (EfficientNet)"""
        print("Evaluating Classification Model (EfficientNet)...")

        # Load model
        model = EfficientNetClassifier(num_classes=7).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        # ROC-AUC
        try:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except Exception:
            auc_score = 0.0

        # Store results
        self.results['classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

        # Print results
        print(f"Classification Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC-ROC: {auc_score:.4f}")

        # Confusion Matrix
        self._plot_confusion_matrix(all_labels, all_preds, class_names)

        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        return self.results['classification']

    def evaluate_segmentation(self, model_path, test_loader):
        """Comprehensive segmentation evaluation"""
        print("Evaluating Segmentation Model...")

        # Load model
        model = UNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        dice_scores = []
        iou_scores = []
        pixel_accuracies = []

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                preds = model(images)
                preds = torch.sigmoid(preds) > 0.5

                # Calculate metrics
                dice = self._dice_coefficient(preds, masks)
                iou = self._iou_score(preds, masks)
                pixel_acc = self._pixel_accuracy(preds, masks)

                dice_scores.append(dice.item())
                iou_scores.append(iou.item())
                pixel_accuracies.append(pixel_acc.item())

        # Calculate averages
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_pixel_acc = np.mean(pixel_accuracies)

        # Store results
        self.results['segmentation'] = {
            'dice_score': avg_dice,
            'iou_score': avg_iou,
            'pixel_accuracy': avg_pixel_acc,
            'dice_std': np.std(dice_scores),
            'iou_std': np.std(iou_scores)
        }

        # Print results
        print(f"Segmentation Results:")
        print(f"   Dice Score: {avg_dice:.4f} +/- {np.std(dice_scores):.4f}")
        print(f"   IoU Score: {avg_iou:.4f} +/- {np.std(iou_scores):.4f}")
        print(f"   Pixel Accuracy: {avg_pixel_acc:.4f} +/- {np.std(pixel_accuracies):.4f}")

        # Plot metrics distribution
        self._plot_segmentation_metrics(dice_scores, iou_scores, pixel_accuracies)

        return self.results['segmentation']

    def _dice_coefficient(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def _iou_score(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def _pixel_accuracy(self, pred, target):
        correct = (pred == target).float().sum()
        total = target.numel()
        return correct / total

    def _plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_roc_curve(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_segmentation_metrics(self, dice_scores, iou_scores, pixel_accuracies):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Dice scores
        axes[0].hist(dice_scores, bins=20, alpha=0.7, color='blue')
        axes[0].set_title('Dice Score Distribution')
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(np.mean(dice_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(dice_scores):.3f}')
        axes[0].legend()

        # IoU scores
        axes[1].hist(iou_scores, bins=20, alpha=0.7, color='green')
        axes[1].set_title('IoU Score Distribution')
        axes[1].set_xlabel('IoU Score')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(np.mean(iou_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(iou_scores):.3f}')
        axes[1].legend()

        # Pixel accuracies
        axes[2].hist(pixel_accuracies, bins=20, alpha=0.7, color='orange')
        axes[2].set_title('Pixel Accuracy Distribution')
        axes[2].set_xlabel('Pixel Accuracy')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(np.mean(pixel_accuracies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(pixel_accuracies):.3f}')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig('segmentation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SKIN DISEASE PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("EfficientNet (Classification)\n")
            f.write("U-Net (Segmentation)\n")
            f.write("=" * 60 + "\n\n")

            if 'classification' in self.results:
                f.write("CLASSIFICATION MODEL RESULTS:\n")
                f.write("-" * 30 + "\n")
                cls_results = self.results['classification']
                f.write(f"Accuracy: {cls_results['accuracy']:.4f}\n")
                f.write(f"Precision: {cls_results['precision']:.4f}\n")
                f.write(f"Recall: {cls_results['recall']:.4f}\n")
                f.write(f"F1-Score: {cls_results['f1_score']:.4f}\n")
                f.write(f"AUC-ROC: {cls_results['auc_score']:.4f}\n\n")

            if 'segmentation' in self.results:
                f.write("SEGMENTATION MODEL RESULTS:\n")
                f.write("-" * 30 + "\n")
                seg_results = self.results['segmentation']
                f.write(f"Dice Score: {seg_results['dice_score']:.4f} +/- {seg_results['dice_std']:.4f}\n")
                f.write(f"IoU Score: {seg_results['iou_score']:.4f} +/- {seg_results['iou_std']:.4f}\n")
                f.write(f"Pixel Accuracy: {seg_results['pixel_accuracy']:.4f}\n\n")

        print(f"Evaluation report saved to: {save_path}")

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()

    # Define paths
    IMAGE_DIR = "../../data/raw/images"
    MASK_DIR = "../../data/raw/masks"
    CSV_PATH = "../../data/raw/metadata.csv"

    CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    # Classification evaluation
    model_path = "../classification/efficientnet_v5.pth"

    if os.path.exists(model_path):
        print("Starting Classification Evaluation...")
        cls_dataset = ISICClassificationDataset(IMAGE_DIR, CSV_PATH)
        cls_loader = DataLoader(cls_dataset, batch_size=16, shuffle=False)

        evaluator.evaluate_classification(
            model_path, cls_loader, CLASS_NAMES
        )

    # Segmentation evaluation
    if os.path.exists("../segmentation/unet_isic_gpu_safe.pth"):
        print("\nStarting Segmentation Evaluation...")
        seg_dataset = ISICSegmentationDataset(IMAGE_DIR, MASK_DIR, img_size=128)
        seg_loader = DataLoader(seg_dataset, batch_size=8, shuffle=False)

        evaluator.evaluate_segmentation(
            "../segmentation/unet_isic_gpu_safe.pth",
            seg_loader
        )

    # Generate comprehensive report
    evaluator.generate_report()
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()
