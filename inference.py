import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add model paths
sys.path.append('models/classification')
sys.path.append('models/segmentation')

from model import EfficientNetClassifier
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


class SkinDiseasePredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load models
        self.load_models()

        # Define transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.seg_transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def load_models(self):
        """Load trained models (EfficientNet + U-Net)"""
        # Classification: EfficientNet
        cls_path = 'models/classification/efficientnet_v5.pth'

        if os.path.exists(cls_path):
            self.model = EfficientNetClassifier(num_classes=7, backbone="efficientnet_b3").to(self.device)
            self.model.load_state_dict(
                torch.load(cls_path, map_location=self.device)
            )
            self.model.eval()
            print("Classification model loaded (EfficientNet)")
        else:
            self.model = None
            print("Classification model not found")

        # Segmentation model
        if os.path.exists('models/segmentation/unet_isic_gpu_safe.pth'):
            self.seg_model = UNet().to(self.device)
            self.seg_model.load_state_dict(
                torch.load('models/segmentation/unet_isic_gpu_safe.pth',
                          map_location=self.device)
            )
            self.seg_model.eval()
            print("Segmentation model loaded")
        else:
            self.seg_model = None
            print("Segmentation model not found")

    def predict_image(self, image_path):
        """Predict on a single image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        results = {}

        # Classification prediction (EfficientNet)
        if self.model:
            cls_input = self.transform(image=image)['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(cls_input)
                cls_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            cls_pred = int(np.argmax(cls_probs))
            cls_confidence = float(cls_probs[cls_pred])

            results['classification'] = {
                'prediction': CLASS_NAMES[cls_pred],
                'confidence': cls_confidence,
                'probabilities': cls_probs
            }

        # Segmentation prediction
        if self.seg_model:
            seg_input = self.seg_transform(image=image)['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                seg_output = self.seg_model(seg_input)
                seg_mask = torch.sigmoid(seg_output) > 0.5
                seg_mask = seg_mask.cpu().numpy()[0, 0]

            results['segmentation'] = {
                'mask': seg_mask,
                'area_ratio': seg_mask.sum() / seg_mask.size
            }

        # Visualize results
        self.visualize_results(original_image, results, image_path)

        return results

    def visualize_results(self, image, results, image_path):
        """Visualize prediction results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Classification result
        if 'classification' in results:
            cls_result = results['classification']
            axes[1].imshow(image)
            axes[1].set_title(f"Classification: {cls_result['prediction']}\n"
                            f"Confidence: {cls_result['confidence']:.3f}")
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Classification\nModel Not Available',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')

        # Segmentation result
        if 'segmentation' in results:
            seg_result = results['segmentation']
            # Overlay mask on image
            overlay = image.copy()
            mask_resized = cv2.resize(seg_result['mask'].astype(np.uint8),
                                      (image.shape[1], image.shape[0]))
            mask_colored = np.zeros_like(image)
            mask_colored[:, :, 0] = mask_resized * 255  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            axes[2].imshow(overlay)
            axes[2].set_title(f"Segmentation\n"
                            f"Lesion Area: {seg_result['area_ratio']:.1%}")
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'Segmentation\nModel Not Available',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].axis('off')

        plt.tight_layout()

        # Save result
        output_name = f"prediction_{os.path.basename(image_path)}"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Results saved as: {output_name}")

    def batch_predict(self, image_folder):
        """Predict on multiple images in a folder"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_folder)
                              if f.lower().endswith(ext)])

        if not image_files:
            print("No images found in the specified folder")
            return

        print(f"Found {len(image_files)} images to process")

        results_summary = []

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"\nProcessing: {image_file}")

            try:
                result = self.predict_image(image_path)
                result['filename'] = image_file
                results_summary.append(result)

                # Print summary
                if 'classification' in result:
                    cls = result['classification']
                    print(f"   Classification: {cls['prediction']} ({cls['confidence']:.3f})")

                if 'segmentation' in result:
                    seg = result['segmentation']
                    print(f"   Lesion Area: {seg['area_ratio']:.1%}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        # Save summary
        self.save_batch_summary(results_summary)

    def save_batch_summary(self, results):
        """Save batch prediction summary"""
        with open('batch_predictions.txt', 'w') as f:
            f.write("BATCH PREDICTION SUMMARY\n")
            f.write("=" * 40 + "\n\n")

            for result in results:
                f.write(f"File: {result['filename']}\n")

                if 'classification' in result:
                    cls = result['classification']
                    f.write(f"  Classification: {cls['prediction']} ({cls['confidence']:.3f})\n")

                if 'segmentation' in result:
                    seg = result['segmentation']
                    f.write(f"  Lesion Area: {seg['area_ratio']:.1%}\n")

                f.write("\n")

        print("Batch summary saved to: batch_predictions.txt")

def main():
    """Main inference function"""
    predictor = SkinDiseasePredictor()

    print("\nSKIN DISEASE PREDICTION - INFERENCE")
    print("=" * 40)
    print("1. Single image prediction")
    print("2. Batch prediction (folder)")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                predictor.predict_image(image_path)
            else:
                print("Image not found")

        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                predictor.batch_predict(folder_path)
            else:
                print("Folder not found")

        elif choice == '3':
            print("Goodbye!")
            break

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
