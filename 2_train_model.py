"""
Step 2: Train the YOLOv8 Model
Run this AFTER you have labeled your images using LabelImg.
"""

from ultralytics import YOLO
import os

def train_model():
    """Train YOLOv8 on the custom dataset."""
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Dataset configuration file
    data_yaml = r"C:\Users\kadam\Documents\datasets\data.yaml"
    
    # Base model (yolov8n = nano, smallest/fastest)
    # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    base_model = "yolov8n.pt"
    
    # Training parameters
    epochs = 50           # Number of training cycles (50-100 is good for small datasets)
    image_size = 640      # Image size for training
    batch_size = 8        # Reduce if you run out of memory
    
    # ========================================
    # TRAINING
    # ========================================
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  YOLOv8 Training for Detail Box Detection                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Starting training...                                        ║
    ║  This may take 5-30 minutes depending on your hardware.      ║
    ║                                                              ║
    ║  Your trained model will be saved at:                        ║
    ║  runs/detect/train/weights/best.pt                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load pre-trained model (transfer learning)
    model = YOLO(base_model)
    
    # Check device
    import torch
    if torch.cuda.is_available():
        print(f"\n✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("\n⚠️ GPU NOT DETECTED (or PyTorch is CPU-only)")
        print("   Using CPU for training (this will be slower)")
        device = 'cpu'

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        plots=True,           # Generate training plots
        save=True,            # Save checkpoints
        cache=True,           # Cache images for faster training
        patience=20,          # Early stopping patience
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate
        device=device,        # Auto-selected device
        workers=4,            # Number of data loading workers
        project="runs/detect",
        name="train",
        exist_ok=True         # Overwrite existing run
    )
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Training Complete!                                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Your trained model is saved at:                             ║
    ║  runs/detect/train/weights/best.pt                           ║
    ║                                                              ║
    ║  Next step:                                                  ║
    ║  Run: python 3_extract_details.py                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    return results


def validate_model(model_path: str = None):
    """Validate the trained model on the validation set."""
    
    if model_path is None:
        model_path = "runs/detect/train/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model()")
        return
    
    model = YOLO(model_path)
    results = model.val(data=r"C:\Users\kadam\Documents\datasets\data.yaml")
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    
    return results


if __name__ == "__main__":
    # Check if dataset has images
    train_images = r"C:\Users\kadam\Documents\datasets\train\images"
    train_labels = r"C:\Users\kadam\Documents\datasets\train\labels"
    
    if not os.listdir(train_images):
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ERROR: No training images found!                            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Please complete these steps first:                          ║
    ║                                                              ║
    ║  1. Run: python 1_prepare_images.py                          ║
    ║     (converts your PDFs to images)                           ║
    ║                                                              ║
    ║  2. Run: labelImg                                            ║
    ║     - Open folder: train/images                              ║
    ║     - Set save format to YOLO                                ║
    ║     - Draw boxes around "detail_box" areas                   ║
    ║     - Save labels to: train/labels                           ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
    elif not os.listdir(train_labels):
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ERROR: No label files found!                                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  You have images but no labels.                              ║
    ║                                                              ║
    ║  Please label your images using LabelImg:                    ║
    ║                                                              ║
    ║  1. Run: labelImg                                            ║
    ║  2. Open folder: train/images                                ║
    ║  3. Change save format to YOLO (bottom left dropdown)        ║
    ║  4. Set default save dir to: train/labels                    ║
    ║  5. For each image:                                          ║
    ║     - Press 'W' to draw a box                                ║
    ║     - Label it as "detail_box"                               ║
    ║     - Press Ctrl+S to save                                   ║
    ║     - Press 'D' for next image                               ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        # Start training
        train_model()
