"""
Step 4: Train Multi-Label Component Classifier
Trains an EfficientNet model to identify components in engineering drawings.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
LABELS_FILE = r"C:\Users\kadam\Documents\datasets\labels.csv"
COMPONENTS_FILE = r"C:\Users\kadam\Documents\datasets\components.txt"
MODEL_OUTPUT_DIR = r"C:\Users\kadam\Documents\datasets\classifier_model"

# Training Settings
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
VALIDATION_SPLIT = 0.2  # 20% for validation

# Model Choice: 'efficientnet_b0' (fast) or 'efficientnet_b2' (more accurate)
MODEL_NAME = 'efficientnet_b0'

# =========================================================
#  Helper Functions
# =========================================================

def load_components(filepath):
    """Load component names from text file."""
    components = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                components.append(line)
    return sorted(list(set(components)))  # Unique, sorted

def load_labels(filepath, components):
    """Load labels CSV and convert to multi-hot encoding."""
    data = []
    comp_to_idx = {c: i for i, c in enumerate(components)}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
                
            image_name = parts[0]
            labels = parts[1:]
            
            # Convert to multi-hot vector
            label_vector = [0] * len(components)
            for label in labels:
                if label in comp_to_idx:
                    label_vector[comp_to_idx[label]] = 1
                else:
                    print(f"Warning: Unknown component '{label}' in {image_name}")
            
            data.append((image_name, label_vector))
    
    return data

# =========================================================
#  Dataset Class
# =========================================================

class ComponentDataset(Dataset):
    def __init__(self, data, images_folder, transform=None):
        self.data = data
        self.images_folder = images_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name, labels = self.data[idx]
        image_path = os.path.join(self.images_folder, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(labels)

# =========================================================
#  Model Definition
# =========================================================

def create_model(num_classes, model_name='efficientnet_b0'):
    """Create EfficientNet model with custom classifier head."""
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
        num_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Replace classifier with multi-label head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
        # No sigmoid here - using BCEWithLogitsLoss
    )
    
    return model

# =========================================================
#  Training Function
# =========================================================

def train_model(model, train_loader, val_loader, num_epochs, device):
    """Train the model with validation."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model

# =========================================================
#  Main Training Script
# =========================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Component Classifier Training                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load components and labels
    print("\nLoading components...")
    components = load_components(COMPONENTS_FILE)
    print(f"Found {len(components)} component types")
    
    print("\nLoading labels...")
    data = load_labels(LABELS_FILE, components)
    print(f"Found {len(data)} labeled images")
    
    if len(data) < 10:
        print("\n❌ ERROR: Not enough labeled data!")
        print("Please label at least 50-100 images in labels.csv first.")
        return
    
    # Split into train/val
    split_idx = int(len(data) * (1 - VALIDATION_SPLIT))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Training set: {len(train_data)} images")
    print(f"Validation set: {len(val_data)} images")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = ComponentDataset(train_data, IMAGES_FOLDER, train_transform)
    val_dataset = ComponentDataset(val_data, IMAGES_FOLDER, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nCreating {MODEL_NAME} model...")
    model = create_model(len(components), MODEL_NAME)
    model = model.to(device)
    
    # Train
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader, EPOCHS, device)
    
    # Save model and metadata
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'component_classifier.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'components': components,
        'model_name': MODEL_NAME,
        'image_size': IMAGE_SIZE
    }, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print("\nNext step: Run 5_classify_components.py to classify your images!")

if __name__ == "__main__":
    main()
