"""
Step 5: Classify Components and Output to Excel
Runs the trained model on images and exports results.
"""

import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================================================
#  🔧 CONFIGURATION - EDIT THIS SECTION
# =========================================================

# Paths
IMAGES_FOLDER = r"C:\Users\kadam\Documents\datasets\figures"
MODEL_PATH = r"C:\Users\kadam\Documents\datasets\classifier_model\component_classifier.pt"
OUTPUT_EXCEL = r"C:\Users\kadam\Documents\datasets\component_predictions.xlsx"

# Inference Settings
CONFIDENCE_THRESHOLD = 0.5  # Min confidence to include a component (0.0 - 1.0)

# =========================================================
#  Model Loading
# =========================================================

def create_model(num_classes, model_name='efficientnet_b0'):
    """Create EfficientNet model with custom classifier head."""
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=None)
        num_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

def load_model(model_path, device):
    """Load trained model and metadata."""
    checkpoint = torch.load(model_path, map_location=device)
    
    components = checkpoint['components']
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    image_size = checkpoint.get('image_size', 224)
    
    model = create_model(len(components), model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, components, image_size

# =========================================================
#  Inference
# =========================================================

def classify_image(model, image_path, transform, components, device, threshold):
    """Classify a single image and return detected components."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return [], []
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Get components above threshold
    detected = []
    confidences = []
    
    for i, (comp, prob) in enumerate(zip(components, probabilities)):
        if prob >= threshold:
            detected.append(comp)
            confidences.append(float(prob))
    
    return detected, confidences

def classify_folder(images_folder, model, transform, components, device, threshold):
    """Classify all images in a folder."""
    results = []
    
    image_files = list(Path(images_folder).glob('*.png')) + \
                  list(Path(images_folder).glob('*.jpg')) + \
                  list(Path(images_folder).glob('*.jpeg'))
    
    print(f"Found {len(image_files)} images to classify\n")
    
    for i, image_path in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(image_files)}...")
        
        detected, confidences = classify_image(
            model, str(image_path), transform, components, device, threshold
        )
        
        results.append({
            'image': image_path.name,
            'components': ', '.join(detected),
            'num_components': len(detected),
            'confidences': ', '.join([f"{c:.2f}" for c in confidences])
        })
    
    return results

# =========================================================
#  Main Script
# =========================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Component Classification - Inference                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using 4_train_classifier.py")
        return
    
    # Load model
    print("\nLoading model...")
    model, components, image_size = load_model(MODEL_PATH, device)
    print(f"Loaded model with {len(components)} component types")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Classify images
    print(f"\nClassifying images from: {IMAGES_FOLDER}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    results = classify_folder(
        IMAGES_FOLDER, model, transform, components, device, CONFIDENCE_THRESHOLD
    )
    
    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"\n✅ Results saved to: {OUTPUT_EXCEL}")
    print(f"Total images processed: {len(results)}")
    
    # Summary stats
    total_detections = sum(r['num_components'] for r in results)
    avg_per_image = total_detections / len(results) if results else 0
    print(f"Total components detected: {total_detections}")
    print(f"Average components per image: {avg_per_image:.1f}")
    
    print("\nNext step: Review predictions in Excel or run 6_verify_predictions.py")

if __name__ == "__main__":
    main()
