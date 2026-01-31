"""
Step 3: Extract Detail Boxes from PDFs using the Trained Model
Run this after training to extract detail boxes from new drawings.
"""

import pymupdf as fitz  # PyMuPDF
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path
from datetime import datetime

def parse_page_string(page_str: str) -> set:
    """
    Parse a page string like "1,3,5-10" into a set of page numbers (1-based).
    Returns None if 'all' is specified.
    """
    if not page_str or page_str.lower() == "all":
        return None
        
    pages = set()
    parts = page_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                # Handle 100-150 (inclusive)
                pages.update(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range '{part}' ignored.")
        else:
            try:
                pages.add(int(part))
            except ValueError:
                print(f"Warning: Invalid page '{part}' ignored.")
                
    return pages

class DetailBoxExtractor:
    """Extract detail boxes from engineering drawings using trained YOLOv8 model."""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the extractor with a trained model.
        
        Args:
            model_path: Path to the trained model (best.pt)
            confidence_threshold: Minimum confidence for detection (0.0 - 1.0)
        """
        if model_path is None:
            # Fallback path if None passed
            model_path = r"runs\detect\train\weights\best.pt"
            
        # Check standard and nested paths
        if not os.path.exists(model_path):
             nested_path = r"runs\detect\runs\detect\train\weights\best.pt"
             if os.path.exists(nested_path):
                 model_path = nested_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Please train the model first using 2_train_model.py"
            )
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Settings
        self.confidence_threshold = confidence_threshold
        self.detection_dpi = 200         # DPI for AI detection
        self.extraction_dpi = 300        # DPI for high-quality extraction
        
    def extract_from_pdf(self, pdf_path: str, output_folder: str = None, page_range: str = "all") -> list:
        """
        Extract all detail boxes from a PDF with page filtering.
        
        Args:
            pdf_path: Path to the PDF file
            output_folder: Folder to save extracted images
            page_range: String like "1,3,5-10" or "all"
            
        Returns:
            List of extracted image paths
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Parse page filter
        target_pages = parse_page_string(page_range)
        
        pdf_name = Path(pdf_path).stem
        
        if output_folder is None:
            output_folder = os.path.join(
                os.path.dirname(pdf_path),
                f"{pdf_name}_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        os.makedirs(output_folder, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        extracted_images = []
        
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path}")
        print(f"Pages: {len(doc)}")
        print(f"Filter: {page_range}")
        print(f"Conf Threshold: {self.confidence_threshold}")
        print(f"Output to: {output_folder}")
        print(f"{'='*60}\n")
        
        for i, page in enumerate(doc):
            page_num = i + 1  # 1-based index
            
            # Check if page is in target list
            if target_pages is not None and page_num not in target_pages:
                continue
                
            print(f"Scanning Page {page_num}...", end=" ")
            
            # Convert page to image for detection
            zoom = self.detection_dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Save temporary image
            temp_img_path = f"_temp_page_{i}.jpg"
            pix.save(temp_img_path)
            
            # Run detection
            results = self.model.predict(
                temp_img_path,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Process detections
            detections = 0
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                    
                for box_idx, box in enumerate(result.boxes):
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    # Convert image coordinates back to PDF coordinates
                    pdf_x1 = x1 / zoom
                    pdf_y1 = y1 / zoom
                    pdf_x2 = x2 / zoom
                    pdf_y2 = y2 / zoom
                    
                    # Create clip rectangle
                    rect = fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2)
                    
                    # Add padding
                    padding = 10
                    rect = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
                    rect = rect & page.rect  # Intersection
                    
                    # Render high-quality crop
                    crop_pix = page.get_pixmap(dpi=self.extraction_dpi, clip=rect)
                    
                    # Save extracted image
                    output_name = f"page{page_num:03d}_detail{box_idx + 1:02d}_conf{conf:.2f}.png"
                    output_path = os.path.join(output_folder, output_name)
                    crop_pix.save(output_path)
                    
                    extracted_images.append(output_path)
                    detections += 1
            
            os.remove(temp_img_path)
            
            if detections > 0:
                print(f"Found {detections} boxes.")
            else:
                print("No boxes.")
        
        doc.close()
        print(f"\nDone! Extracted {len(extracted_images)} images.")
        return extracted_images
    
    def extract_from_folder(self, pdf_folder: str, output_base: str = None) -> dict:
        """
        Extract detail boxes from all PDFs in a folder.
        
        Args:
            pdf_folder: Folder containing PDF files
            output_base: Base folder for all outputs (default: pdf_folder/extracted)
            
        Returns:
            Dictionary mapping PDF names to their extracted image paths
        """
        if not os.path.exists(pdf_folder):
            raise FileNotFoundError(f"Folder not found: {pdf_folder}")
        
        if output_base is None:
            output_base = os.path.join(pdf_folder, "extracted")
        
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in: {pdf_folder}")
            return {}
        
        print(f"\n{'#'*60}")
        print(f"BATCH PROCESSING")
        print(f"Found {len(pdf_files)} PDF files")
        print(f"{'#'*60}\n")
        
        results = {}
        
        for pdf_path in pdf_files:
            pdf_output = os.path.join(output_base, pdf_path.stem)
            results[pdf_path.name] = self.extract_from_pdf(str(pdf_path), pdf_output)
        
        # Print summary
        total = sum(len(v) for v in results.values())
        print(f"\n{'#'*60}")
        print(f"BATCH COMPLETE!")
        print(f"Processed {len(pdf_files)} PDFs")
        print(f"Extracted {total} total detail boxes")
        print(f"{'#'*60}\n")
        
        return results


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Detail Box Extractor v2.0                                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================
    #  🔧 CONFIGURATION - EDIT THIS SECTION
    # =========================================================
    
    # 1. MODEL PATH
    # We check multiple locations automatically. 
    # Edit if your model is somewhere unique.
    MODEL_PATH = r"C:\Users\kadam\Documents\datasets\runs\detect\runs\detect\train\weights\best.pt"
    
    # 2. INPUT PDF (Choose ONE mechanism)
    # Mechanism A: Single File
    SINGLE_PDF_PATH = r"C:\Users\kadam\Documents\datasets\heat_and_mass_transfr_by_cengel.pdf"
    
    # Mechanism B: Folder of PDFs (Uncomment in code below to use)
    PDF_FOLDER_PATH = r"C:\path\to\your\pdfs"

    # 3. OUTPUT FOLDER
    # If None, it creates a folder next to the PDF
    OUTPUT_FOLDER = r"C:\Users\kadam\Documents\datasets\figures"
    
    # 4. SETTINGS
    CONFIDENCE = 0.5            # Min confidence (0.0 - 1.0). Higher = stricter.
    PAGE_RANGE = "all"          # Options: "all", or "1,2,5", or "100-150"
                                # Example: "2,5,100-150,60"

    # =========================================================
    #  🚀 EXECUTION
    # =========================================================

    # Fix paths if they are relative/broken (Optional check)
    possible_paths = [
        MODEL_PATH,
        r"C:\Users\kadam\Documents\datasets\runs\detect\train\weights\best.pt",
        r"runs\detect\train\weights\best.pt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            MODEL_PATH = p
            break

    try:
        extractor = DetailBoxExtractor(MODEL_PATH, CONFIDENCE)
        
        # Run Extraction
        extractor.extract_from_pdf(
            pdf_path=SINGLE_PDF_PATH, 
            output_folder=OUTPUT_FOLDER,
            page_range=PAGE_RANGE
        )
        
        # To verify folder processing, uncomment:
        # extractor.extract_from_folder(PDF_FOLDER_PATH, OUTPUT_FOLDER)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
