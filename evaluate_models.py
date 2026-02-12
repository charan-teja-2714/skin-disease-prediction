import torch
import os
import sys
import numpy as np
from datetime import datetime

# Add paths
sys.path.append('models/classification')
sys.path.append('models/segmentation')

def check_models():
    """Check if trained models exist"""
    seg_model = 'models/segmentation/unet_isic_gpu_safe.pth'
    cls_model = 'models/classification/efficientnet_best.pth'

    print("Checking for trained models...")
    print(f"Segmentation model: {'Found' if os.path.exists(seg_model) else 'Not found'}")
    print(f"EfficientNet classifier: {'Found' if os.path.exists(cls_model) else 'Not found'}")

    return os.path.exists(seg_model), os.path.exists(cls_model)

def get_model_info():
    """Get actual model information"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    seg_exists, cls_exists = check_models()

    # Segmentation model info
    if seg_exists:
        try:
            from unet import UNet
            model = UNet()
            state_dict = torch.load('models/segmentation/unet_isic_gpu_safe.pth', map_location='cpu')
            model.load_state_dict(state_dict)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size = os.path.getsize('models/segmentation/unet_isic_gpu_safe.pth') / (1024 * 1024)

            results['segmentation_model'] = {
                'status': 'loaded',
                'architecture': 'U-Net (4-level encoder-decoder)',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': round(model_size, 2)
            }
            print(f"\nSegmentation Model Loaded")
            print(f"   Parameters: {total_params:,}")
            print(f"   Size: {model_size:.2f} MB")
        except Exception as e:
            results['segmentation_model'] = {'status': 'error', 'message': str(e)}
            print(f"Error loading segmentation model: {e}")
    else:
        results['segmentation_model'] = {'status': 'not_found'}

    # Classification model info
    if cls_exists:
        try:
            from model import EfficientNetClassifier
            model = EfficientNetClassifier(num_classes=7)
            state_dict = torch.load('models/classification/efficientnet_best.pth', map_location='cpu')
            model.load_state_dict(state_dict)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size = os.path.getsize('models/classification/efficientnet_best.pth') / (1024 * 1024)

            results['classification_model'] = {
                'status': 'loaded',
                'architecture': 'EfficientNet-B0 (timm)',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': round(model_size, 2),
                'num_classes': 7,
                'class_names': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
            }
            print(f"\nClassification Model Loaded")
            print(f"   Parameters: {total_params:,}")
            print(f"   Size: {model_size:.2f} MB")
        except Exception as e:
            results['classification_model'] = {'status': 'error', 'message': str(e)}
            print(f"Error loading classification model: {e}")
    else:
        results['classification_model'] = {'status': 'not_found'}

    return results

def test_single_inference():
    """Test inference speed with dummy data"""
    print("\nTesting Inference Speed...")

    seg_exists, cls_exists = check_models()
    results = {}

    # Test segmentation
    if seg_exists:
        try:
            from unet import UNet
            model = UNet()
            state_dict = torch.load('models/segmentation/unet_isic_gpu_safe.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            dummy_input = torch.randn(1, 3, 128, 128)

            with torch.no_grad():
                _ = model(dummy_input)

            import time
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            end = time.time()

            avg_time = (end - start) / 10 * 1000
            results['segmentation_inference_ms'] = round(avg_time, 2)
            print(f"   Segmentation: {avg_time:.2f} ms/image")
        except Exception as e:
            print(f"   Segmentation: Error - {e}")

    # Test classification
    if cls_exists:
        try:
            from model import EfficientNetClassifier
            model = EfficientNetClassifier(num_classes=7)
            state_dict = torch.load('models/classification/efficientnet_best.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            dummy_input = torch.randn(1, 3, 224, 224)

            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)

            import time
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            end = time.time()

            avg_time = (end - start) / 10 * 1000
            results['classification_inference_ms'] = round(avg_time, 2)
            print(f"   Classification (EfficientNet): {avg_time:.2f} ms/image")
        except Exception as e:
            print(f"   Classification: Error - {e}")

    return results

def save_results(results):
    """Save results to file"""
    import json

    output_file = f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Also create a readable text report
    report_file = f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SKIN CANCER PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write(f"Device: {results['device']}\n\n")

        # Segmentation model
        f.write("SEGMENTATION MODEL (U-Net):\n")
        f.write("-" * 40 + "\n")
        if results['segmentation_model']['status'] == 'loaded':
            seg = results['segmentation_model']
            f.write(f"Status: Loaded\n")
            f.write(f"Architecture: {seg['architecture']}\n")
            f.write(f"Parameters: {seg['total_parameters']:,}\n")
            f.write(f"Model Size: {seg['model_size_mb']} MB\n")
            if 'segmentation_inference_ms' in results:
                f.write(f"Inference Time: {results['segmentation_inference_ms']} ms\n")
        else:
            f.write(f"Status: {results['segmentation_model']['status']}\n")
        f.write("\n")

        # Classification model
        f.write("CLASSIFICATION MODEL (EfficientNet):\n")
        f.write("-" * 40 + "\n")
        if results['classification_model']['status'] == 'loaded':
            cls = results['classification_model']
            f.write(f"Status: Loaded\n")
            f.write(f"Architecture: {cls['architecture']}\n")
            f.write(f"Parameters: {cls['total_parameters']:,}\n")
            f.write(f"Model Size: {cls['model_size_mb']} MB\n")
            f.write(f"Number of Classes: {cls['num_classes']}\n")
            f.write(f"Classes: {', '.join(cls['class_names'])}\n")
            if 'classification_inference_ms' in results:
                f.write(f"Inference Time: {results['classification_inference_ms']} ms\n")
        else:
            f.write(f"Status: {results['classification_model']['status']}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("NOTE: To get accuracy metrics, run:\n")
        f.write("  cd models/classification && python evaluate.py\n")
        f.write("=" * 60 + "\n")

    print(f"Report saved to: {report_file}")

def main():
    """Main function"""
    print("=" * 60)
    print("SKIN CANCER PREDICTION - MODEL EVALUATION")
    print("=" * 60)

    # Get model info
    results = get_model_info()

    # Test inference
    inference_results = test_single_inference()
    results.update(inference_results)

    # Save results
    save_results(results)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    print("\nSUMMARY:")
    print(f"   Device: {results['device']}")

    if results['segmentation_model']['status'] == 'loaded':
        seg = results['segmentation_model']
        print(f"   Segmentation Model: {seg['model_size_mb']} MB, {seg['total_parameters']:,} params")
        if 'segmentation_inference_ms' in results:
            print(f"   Segmentation Speed: {results['segmentation_inference_ms']} ms")

    if results['classification_model']['status'] == 'loaded':
        cls = results['classification_model']
        print(f"   Classification Model: {cls['model_size_mb']} MB, {cls['total_parameters']:,} params")
        if 'classification_inference_ms' in results:
            print(f"   Classification Speed: {results['classification_inference_ms']} ms")

if __name__ == "__main__":
    main()
