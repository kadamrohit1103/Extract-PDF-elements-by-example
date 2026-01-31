"""
Step 1: Convert PDF pages to images for labeling
This script converts PDF pages to JPG images that you'll label using LabelImg.
"""

import pymupdf as fitz  # PyMuPDF
import os
from pathlib import Path

def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 200):
    """
    Convert all pages of a PDF to JPG images.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save the images
        dpi: Resolution for the output images (200 is good for training)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    
    print(f"Converting {len(doc)} pages from: {pdf_path}")
    
    for page_num, page in enumerate(doc):
        # Render page at specified DPI
        zoom = dpi / 72  # 72 is the default DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as JPG
        output_path = os.path.join(output_folder, f"{pdf_name}_page{page_num + 1:03d}.jpg")
        pix.save(output_path)
        print(f"  Saved: {output_path}")
    
    print(f"\nDone! Converted {len(doc)} pages to {output_folder}")
    doc.close()


def main():
    # ========================================
    # CONFIGURE THESE PATHS
    # ========================================
    
    # Option 1: Convert a single PDF
    pdf_file = r"C:\Users\kadam\Documents\datasets\heat_and_mass_transfr_by_cengel.pdf"
    
    # Option 2: Convert all PDFs in a folder
    #pdf_folder = r"C:\path\to\your\pdfs"
    
    # Output folder for training images
    output_folder = r"C:\Users\kadam\Documents\datasets\all_pages"
    
    # ========================================
    # CHOOSE ONE MODE AND UNCOMMENT
    # ========================================
    
    # Mode 1: Single PDF
    pdf_to_images(pdf_file, output_folder)
    
    # Mode 2: All PDFs in a folder (first 5-6 for training)
    # for pdf in Path(pdf_folder).glob("*.pdf"):
    #     pdf_to_images(str(pdf), output_folder)
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  PDF to Image Converter for YOLOv8 Training                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  INSTRUCTIONS:                                               ║
    ║                                                              ║
    ║  1. Edit this script and set:                                ║
    ║     - pdf_file or pdf_folder path                            ║
    ║     - output_folder (default is train/images)                ║
    ║                                                              ║
    ║  2. Uncomment ONE of the modes above                         ║
    ║                                                              ║
    ║  3. Run: python 1_prepare_images.py                          ║
    ║                                                              ║
    ║  4. After conversion, run LabelImg to annotate:              ║
    ║     > labelImg                                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
