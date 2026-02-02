# YOLOv8 Detail Box Extractor

AI-powered tool to automatically detect and extract detail boxes from engineering drawings.

## 📁 Project Structure

```
datasets/
├── train/
│   ├── images/     # Training images (JPG/PNG)
│   └── labels/     # YOLO format labels (.txt)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
├── data.yaml                 # Dataset configuration
├── requirements.txt          # Python dependencies
├── 1_prepare_images.py       # Convert PDFs to images
├── 2_train_model.py          # Train the YOLOv8 model
└── 3_extract_details.py      # Extract details from new PDFs
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```powershell
cd C:\Users\kadam\Documents\datasets
pip install -r requirements.txt
```

### Step 2: Select Training Images

⚠️ **CRITICAL: Do NOT train on all 800+ pages!**

1. Go to the `all_pages` folder.
2. **Copy** 5 to 6 distinct images that have "Detail Boxes".
3. **Paste** them into `train/images`.
   - `train/images` should ONLY contain the 5-6 images you want to label.

### Step 3: Label Your Images

1. Run the launcher script (fixes "command not found" error):
   ```powershell
   python run_labelImg.py
   ```

2. In LabelImg:
   - Click **Open Dir** → Select `train/images`
   - Click **Change Save Dir** → Select `train/labels`
   - **Important**: Click the format button (bottom left) and select **YOLO**

3. For each of your 5-6 images:
   - Press **W** to draw a bounding box
   - Draw a box around each "Detail Box"
   - Label it as `detail_box`
   - Press **Ctrl+S** to save
   - Press **D** for next image

### Step 4: Train the Model

```powershell
python 2_train_model.py
```

Training takes 5-30 minutes. Your model will be saved to:
`runs/detect/train/weights/best.pt`

### Step 5: Extract Details from New PDFs

1. Edit `3_extract_details.py` and set your PDF path
2. Run:
   ```powershell
   python 3_extract_details.py
   ```

## ⚙️ Configuration

### Confidence Threshold

In `3_extract_details.py`, adjust `confidence_threshold`:
- **0.5** (default): Balanced detection
- **0.3**: More detections (may include false positives)
- **0.7**: Fewer but more confident detections

### Extraction Quality

Adjust `extraction_dpi` for output quality:
- **200**: Normal quality, smaller files
- **300** (default): High quality
- **400**: Very high quality, larger files

## 📝 Label Format (YOLO)

Each `.txt` file in `train/labels` contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1). Example:
```
0 0.5 0.5 0.2 0.15
```

## 🔧 Troubleshooting

### "No module named 'ultralytics'"
```powershell
pip install ultralytics
```

### "Model not found"
Make sure training completed successfully. Check `runs/detect/train/weights/` for `best.pt`.

### Low detection accuracy
- Add more training images (10-20 is better than 5-6)
- Ensure consistent labeling
- Increase training epochs to 100

### Out of memory during training
Reduce `batch_size` in `2_train_model.py` (try 4 or 2).

## 💡 Tips

1. **Variety matters**: Include detail boxes of different sizes and positions
2. **Be consistent**: Label ALL detail boxes in each training image
3. **Validation set**: Copy 1-2 labeled images to `val/` for better training
4. **GPU acceleration**: For faster training, install PyTorch with CUDA support

---

## 🔬 Component Classifier (Phase 2)

After extracting detail images, you can train a classifier to identify which components are in each image.

### Files

| File | Purpose |
|------|---------|
| `components.txt` | List of component names (edit as needed) |
| `labels.csv` | Your training labels |
| `4_train_classifier.py` | Train the classifier |
| `5_classify_components.py` | Run inference → Excel output |
| `6_verify_predictions.py` | GUI to review/correct predictions |

### Workflow

#### Step 1: Prepare Labels

1. Review `components.txt` - add/remove component names as needed
2. Open `labels.csv` and add training data:
   ```
   image_filename.png,Component1,Component2,Component3
   ```
   Example:
   ```
   page001_detail01.png,Base Plate,Guide/Hold-on
   page002_detail01.png,Trunnion,Wear Pad,Cap Plate
   page003_detail02.png,Pipe Shoe
   ```
3. Label **100-150 images** minimum for good accuracy

#### Step 2: Train the Model

```powershell
python 4_train_classifier.py
```

- Takes ~10-30 min (CPU) or ~5 min (GPU)
- Model saves to `classifier_model/component_classifier.pt`

#### Step 3: Classify All Images

```powershell
python 5_classify_components.py
```

- Processes all images in `figures/` folder
- Outputs `component_predictions.xlsx`

#### Step 4: Verify & Improve (Optional)

```powershell
python 6_verify_predictions.py
```

- Opens GUI showing each image + prediction
- Click **Accept** if correct, or check boxes + **Save Corrections**
- Click **Save All** to add verified labels to `labels.csv`
- Retrain for better accuracy

### Configuration

Edit the top of each script to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `IMAGES_FOLDER` | `figures/` | Where extracted images are |
| `CONFIDENCE_THRESHOLD` | `0.5` | Min confidence to include component |
| `EPOCHS` | `30` | Training iterations |
| `MODEL_NAME` | `efficientnet_b0` | Use `efficientnet_b2` for better accuracy |

### Tips for Better Accuracy

1. **More labels = better accuracy** (aim for 50+ per component type)
2. **Be consistent** with component names (must match `components.txt` exactly)
3. **Include variations** - different sizes, orientations, contexts
4. **Iterate**: Train → Verify → Add corrections → Retrain
