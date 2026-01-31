"""
Quick Test Script
Run this to verify your installation is working correctly.
"""

import sys

def check_installation():
    """Check if all required packages are installed."""
    
    print("=" * 50)
    print("YOLOv8 Detail Box Extractor - Installation Check")
    print("=" * 50)
    print()
    
    packages = {
        'ultralytics': 'YOLOv8 (Machine Learning)',
        'pymupdf': 'PyMuPDF (PDF Processing)',
        'PIL': 'Pillow (Image Processing)',
    }
    
    all_ok = True
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {description}: Installed")
        except ImportError:
            print(f"❌ {description}: NOT INSTALLED")
            all_ok = False
    
    # Check for GPU
    print()
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU Available: {gpu_name}")
            print("   Training will be FAST!")
        else:
            print("💻 No GPU detected - Training will use CPU (slower but works)")
    except ImportError:
        print("💻 PyTorch not installed - Training will use CPU")
    
    print()
    print("=" * 50)
    
    if all_ok:
        print("✅ All required packages are installed!")
        print()
        print("Next steps:")
        print("  1. Run: python 1_prepare_images.py")
        print("  2. Run: labelImg")
        print("  3. Run: python 2_train_model.py")
        print("  4. Run: python 3_extract_details.py")
    else:
        print("❌ Some packages are missing!")
        print()
        print("Install them with:")
        print("  pip install -r requirements.txt")
    
    print("=" * 50)
    
    return all_ok


if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
