import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_demo_outputs():
    """Create demo outputs showing all system capabilities"""
    
    print("🏥 SKIN CANCER PREDICTION SYSTEM - DEMO OUTPUTS")
    print("=" * 60)
    
    # Demo data (simulated results)
    demo_results = {
        "image_path": "demo_lesion.jpg",
        "timestamp": datetime.now().isoformat(),
        
        # 1. LESION SEGMENTATION MASK
        "segmentation": {
            "lesion_area_ratio": 0.23,
            "segmentation_confidence": 0.89,
            "lesion_area_pixels": 15420,
            "total_area_pixels": 67000
        },
        
        # 2. MULTI-CLASS SKIN DISEASE PREDICTION
        "classification": {
            "predicted_class": "Melanoma (MEL)",
            "predicted_class_index": 0,
            "prediction_confidence": 0.87,
            "is_malignant": True,
            "malignancy_probability": 0.91,
            "all_class_probabilities": {
                "Melanoma (MEL)": 0.87,
                "Melanocytic nevus (NV)": 0.05,
                "Basal cell carcinoma (BCC)": 0.04,
                "Actinic keratosis (AKIEC)": 0.02,
                "Benign keratosis (BKL)": 0.01,
                "Dermatofibroma (DF)": 0.01,
                "Vascular lesion (VASC)": 0.00
            }
        },
        
        # 3. ABCDE-BASED CLINICAL EXPLANATION
        "abcde_analysis": {
            "asymmetry_score": 0.72,
            "border_irregularity": 0.68,
            "color_variation": 0.85,
            "diameter_mm": 8.3,
            "evolution_risk": 0.74,
            "overall_abcde_score": 0.71,
            "clinical_interpretation": [
                "High asymmetry detected - lesion is not symmetrical",
                "Highly irregular borders detected", 
                "High color variation - multiple colors present",
                "Large diameter (8.3mm) - exceeds 6mm threshold",
                "High evolution risk - multiple concerning features"
            ]
        },
        
        # 4. SKIN TONE ANALYSIS FOR FAIRNESS
        "skin_tone": {
            "estimated_skin_tone": "medium",
            "fairness_group": "medium",
            "note": "Skin tone estimation for bias assessment in AI predictions"
        },
        
        # 5. RISK ASSESSMENT
        "risk_assessment": {
            "overall_risk_score": 0.83,
            "risk_level": "HIGH",
            "risk_factors": [
                "Classified as Melanoma (MEL)",
                "High asymmetry detected",
                "Irregular borders detected",
                "High color variation",
                "Large diameter (8.3mm)",
                "Evolution indicators present"
            ],
            "recommendation": "CRITICAL: Immediate medical attention required. Possible melanoma detected."
        }
    }
    
    # Create output directory
    output_dir = f"demo_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    create_demo_visualizations(demo_results, output_dir)
    
    # Generate reports
    create_demo_reports(demo_results, output_dir)
    
    # Show fairness metrics across different skin tones
    create_fairness_demo(output_dir)
    
    print(f"\n📊 Demo outputs created in: {output_dir}/")
    print(f"📁 Files generated:")
    print(f"   ✅ comprehensive_analysis.png - Main visualization")
    print(f"   ✅ segmentation_mask.png - Lesion mask")
    print(f"   ✅ clinical_report.txt - Detailed medical report")
    print(f"   ✅ fairness_analysis.png - Bias assessment")
    print(f"   ✅ demo_results.json - Raw prediction data")
    
    return output_dir

def create_demo_visualizations(results, output_dir):
    """Create comprehensive demo visualizations"""
    
    # Create synthetic lesion image and mask for demo
    np.random.seed(42)
    demo_image = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
    demo_mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Create irregular lesion shape
    center = (100, 100)
    for angle in np.linspace(0, 2*np.pi, 100):
        radius = 40 + 15 * np.sin(5*angle) + 10 * np.random.random()
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        if 0 <= x < 200 and 0 <= y < 200:
            demo_mask[max(0, y-3):min(200, y+3), max(0, x-3):min(200, x+3)] = 255
    
    # Main comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Skin Cancer Analysis - DEMO', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(demo_image)
    axes[0, 0].set_title('Original Lesion Image')
    axes[0, 0].axis('off')
    
    # 2. Segmentation Mask
    axes[0, 1].imshow(demo_image)
    axes[0, 1].imshow(demo_mask, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title(f'Lesion Segmentation\nArea: {results["segmentation"]["lesion_area_ratio"]:.1%}')
    axes[0, 1].axis('off')
    
    # 3. Classification Results
    cls_results = results['classification']
    class_names = list(cls_results['all_class_probabilities'].keys())
    class_probs = list(cls_results['all_class_probabilities'].values())
    
    colors = ['red' if 'Melanoma' in name or 'carcinoma' in name or 'keratosis' in name 
              else 'green' for name in class_names]
    
    bars = axes[0, 2].barh([name.split('(')[0].strip() for name in class_names], class_probs, color=colors)
    axes[0, 2].set_title('Multi-class Predictions')
    axes[0, 2].set_xlabel('Probability')
    
    # 4. ABCDE Analysis
    abcde = results['abcde_analysis']
    abcde_scores = [
        abcde['asymmetry_score'],
        abcde['border_irregularity'],
        abcde['color_variation'],
        min(abcde['diameter_mm'] / 10, 1.0),
        abcde['evolution_risk']
    ]
    abcde_labels = ['Asymmetry', 'Border', 'Color', 'Diameter', 'Evolution']
    
    bar_colors = ['red' if s > 0.6 else 'orange' if s > 0.4 else 'green' for s in abcde_scores]
    axes[1, 0].bar(abcde_labels, abcde_scores, color=bar_colors)
    axes[1, 0].set_title('ABCDE Clinical Analysis')
    axes[1, 0].set_ylabel('Risk Score')
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Risk Assessment
    risk = results['risk_assessment']
    risk_colors = {'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'yellow', 'MINIMAL': 'green'}
    
    axes[1, 1].pie([risk['overall_risk_score'], 1-risk['overall_risk_score']], 
                  labels=[f'{risk["risk_level"]} RISK', 'Safe'], 
                  colors=[risk_colors.get(risk['risk_level'], 'gray'), 'lightgray'],
                  autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title(f'Overall Risk Assessment\nScore: {risk["overall_risk_score"]:.2f}')
    
    # 6. Summary Information
    axes[1, 2].axis('off')
    summary_text = f"""
PREDICTION SUMMARY

Predicted Class:
{cls_results['predicted_class']}

Confidence: {cls_results['prediction_confidence']:.1%}

Malignancy Risk: {cls_results['malignancy_probability']:.1%}

Skin Tone: {results['skin_tone']['estimated_skin_tone']}

Key Risk Factors:
{chr(10).join(['• ' + factor for factor in risk['risk_factors'][:3]])}

Recommendation:
{risk['recommendation'][:80]}...
    """
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual segmentation mask
    plt.figure(figsize=(8, 8))
    plt.imshow(demo_mask, cmap='gray')
    plt.title('Lesion Segmentation Mask')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'segmentation_mask.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_demo_reports(results, output_dir):
    """Create detailed demo reports"""
    
    # Clinical report
    report_path = os.path.join(output_dir, 'clinical_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE SKIN CANCER ANALYSIS REPORT - DEMO\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {results['timestamp']}\n")
        f.write(f"Image: {results['image_path']}\n\n")
        
        # Classification Results
        cls = results['classification']
        f.write("1. CLASSIFICATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Predicted Diagnosis: {cls['predicted_class']}\n")
        f.write(f"Confidence: {cls['prediction_confidence']:.1%}\n")
        f.write(f"Malignancy Probability: {cls['malignancy_probability']:.1%}\n")
        f.write(f"Classification: {'MALIGNANT' if cls['is_malignant'] else 'BENIGN'}\n\n")
        
        f.write("All Class Probabilities:\n")
        for class_name, prob in cls['all_class_probabilities'].items():
            f.write(f"  {class_name}: {prob:.1%}\n")
        f.write("\n")
        
        # Segmentation Results
        seg = results['segmentation']
        f.write("2. LESION SEGMENTATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Lesion Area: {seg['lesion_area_ratio']:.1%} of image\n")
        f.write(f"Segmentation Confidence: {seg['segmentation_confidence']:.1%}\n")
        f.write(f"Lesion Area (pixels): {seg['lesion_area_pixels']:,}\n\n")
        
        # ABCDE Analysis
        abcde = results['abcde_analysis']
        f.write("3. ABCDE CLINICAL ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"A - Asymmetry Score: {abcde['asymmetry_score']:.3f}\n")
        f.write(f"B - Border Irregularity: {abcde['border_irregularity']:.3f}\n")
        f.write(f"C - Color Variation: {abcde['color_variation']:.3f}\n")
        f.write(f"D - Diameter: {abcde['diameter_mm']:.1f} mm\n")
        f.write(f"E - Evolution Risk: {abcde['evolution_risk']:.3f}\n")
        f.write(f"Overall ABCDE Score: {abcde['overall_abcde_score']:.3f}\n\n")
        
        f.write("Clinical Interpretation:\n")
        for interpretation in abcde['clinical_interpretation']:
            f.write(f"  • {interpretation}\n")
        f.write("\n")
        
        # Risk Assessment
        risk = results['risk_assessment']
        f.write("4. RISK ASSESSMENT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Risk Level: {risk['risk_level']}\n")
        f.write(f"Risk Score: {risk['overall_risk_score']:.3f}\n\n")
        
        f.write("Risk Factors Identified:\n")
        for factor in risk['risk_factors']:
            f.write(f"  • {factor}\n")
        f.write("\n")
        
        f.write("CLINICAL RECOMMENDATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{risk['recommendation']}\n\n")
        
        # Fairness Information
        skin_tone = results['skin_tone']
        f.write("5. FAIRNESS & BIAS ASSESSMENT:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Estimated Skin Tone: {skin_tone['estimated_skin_tone']}\n")
        f.write(f"Fairness Group: {skin_tone['fairness_group']}\n")
        f.write("Note: This analysis includes skin tone estimation to assess\n")
        f.write("potential algorithmic bias across different demographic groups.\n\n")
        
        f.write("DISCLAIMER:\n")
        f.write("-" * 40 + "\n")
        f.write("This AI analysis is for research and educational purposes only.\n")
        f.write("It should not replace professional medical diagnosis.\n")
        f.write("Always consult a qualified dermatologist for medical advice.\n")
    
    # Save JSON results
    import json
    with open(os.path.join(output_dir, 'demo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def create_fairness_demo(output_dir):
    """Create fairness analysis demo"""
    
    # Demo fairness data across skin tones
    fairness_data = {
        'light': {
            'total_cases': 45,
            'malignant_rate': 0.22,
            'avg_confidence': 0.89,
            'high_risk_rate': 0.18
        },
        'medium': {
            'total_cases': 38,
            'malignant_rate': 0.26,
            'avg_confidence': 0.85,
            'high_risk_rate': 0.21
        },
        'dark': {
            'total_cases': 17,
            'malignant_rate': 0.29,
            'avg_confidence': 0.81,
            'high_risk_rate': 0.24
        }
    }
    
    # Create fairness visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fairness Analysis Across Skin Tones - DEMO', fontsize=16, fontweight='bold')
    
    skin_tones = list(fairness_data.keys())
    
    # Malignant detection rate by skin tone
    malignant_rates = [fairness_data[tone]['malignant_rate'] for tone in skin_tones]
    axes[0, 0].bar(skin_tones, malignant_rates, color='red', alpha=0.7)
    axes[0, 0].set_title('Malignant Detection Rate by Skin Tone')
    axes[0, 0].set_ylabel('Malignant Rate')
    
    # Average confidence by skin tone
    confidences = [fairness_data[tone]['avg_confidence'] for tone in skin_tones]
    axes[0, 1].bar(skin_tones, confidences, color='blue', alpha=0.7)
    axes[0, 1].set_title('Average Prediction Confidence by Skin Tone')
    axes[0, 1].set_ylabel('Average Confidence')
    
    # High risk rate by skin tone
    risk_rates = [fairness_data[tone]['high_risk_rate'] for tone in skin_tones]
    axes[1, 0].bar(skin_tones, risk_rates, color='orange', alpha=0.7)
    axes[1, 0].set_title('High Risk Detection Rate by Skin Tone')
    axes[1, 0].set_ylabel('High Risk Rate')
    
    # Sample size by skin tone
    sample_sizes = [fairness_data[tone]['total_cases'] for tone in skin_tones]
    axes[1, 1].bar(skin_tones, sample_sizes, color='green', alpha=0.7)
    axes[1, 1].set_title('Sample Size by Skin Tone')
    axes[1, 1].set_ylabel('Number of Cases')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save fairness report
    with open(os.path.join(output_dir, 'fairness_report.txt'), 'w') as f:
        f.write("FAIRNESS ANALYSIS REPORT - DEMO\n")
        f.write("=" * 40 + "\n\n")
        
        for tone, data in fairness_data.items():
            f.write(f"Skin Tone: {tone.capitalize()}\n")
            f.write(f"  Sample Size: {data['total_cases']}\n")
            f.write(f"  Malignant Detection Rate: {data['malignant_rate']:.1%}\n")
            f.write(f"  Average Confidence: {data['avg_confidence']:.1%}\n")
            f.write(f"  High Risk Rate: {data['high_risk_rate']:.1%}\n\n")
        
        # Calculate fairness metrics
        malignant_rates = [data['malignant_rate'] for data in fairness_data.values()]
        confidence_rates = [data['avg_confidence'] for data in fairness_data.values()]
        
        f.write("FAIRNESS METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Malignant Rate Std Dev: {np.std(malignant_rates):.3f}\n")
        f.write(f"Confidence Std Dev: {np.std(confidence_rates):.3f}\n")
        f.write(f"Max Malignant Rate Difference: {np.max(malignant_rates) - np.min(malignant_rates):.3f}\n")
        f.write("\nNote: Lower standard deviation indicates more fair performance across groups.\n")

def main():
    """Main demo function"""
    print("🚀 Creating comprehensive demo outputs...")
    
    output_dir = create_demo_outputs()
    
    print("\n🎉 Demo completed successfully!")
    print("\n📋 SYSTEM OUTPUTS DEMONSTRATED:")
    print("   ✅ 1. Lesion segmentation mask")
    print("   ✅ 2. Multi-class skin disease prediction")
    print("   ✅ 3. Prediction confidence scores")
    print("   ✅ 4. ABCDE-based clinical explanation")
    print("   ✅ 5. Fairness metrics across skin-tone groups")
    
    print(f"\n📁 All demo files saved in: {output_dir}/")
    print("\n🔍 To test with real images, use:")
    print("   python comprehensive_predictor.py")
    print("   python batch_evaluator.py")

if __name__ == "__main__":
    main()