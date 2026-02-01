"""
Step 1: Convert PDF pages to images for labeling
This script converts PDF pages to JPG images that you'll label using LabelImg.
"""

import pymupdf as fitz  # PyMuPDF
import os
from pathlib import Path

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
                pages.update(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range '{part}' ignored.")
        else:
            try:
                pages.add(int(part))
            except ValueError:
                print(f"Warning: Invalid page '{part}' ignored.")
                
    return pages

def pdf_to_images(pdf_path: str, output_folder: str, page_range: str = "all", dpi: int = 200):
    """
    Convert pages of a PDF to JPG images.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save the images
        page_range: String page filter ("1-5, 10")
        dpi: Resolution for the output images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    target_pages = parse_page_string(page_range)
    
    pdf_name = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    
    print(f"Processing: {pdf_path}")
    print(f"Total Pages: {len(doc)}")
    print(f"Filter: {page_range}")
    
    count = 0
    for i, page in enumerate(doc):
        page_num = i + 1
        
        # Check filter
        if target_pages is not None and page_num not in target_pages:
            continue
            
        # Render page
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as JPG
        output_path = os.path.join(output_folder, f"{pdf_name}_page{page_num:03d}.jpg")
        pix.save(output_path)
        print(f"  Saved: {output_path}")
        count += 1
    
    print(f"Converted {count} pages from {pdf_name}\n")
    doc.close()


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  PDF to Image Converter v2.0                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================
    #  🔧 CONFIGURATION - EDIT THIS SECTION
    # =========================================================
    
    # 1. INPUT PDF (Choose ONE mechanism)
    # Mechanism A: Single File
    SINGLE_PDF_PATH = r"C:\Users\kadam\Documents\datasets\heat_and_mass_transfr_by_cengel.pdf"
    
    # Mechanism B: Folder of PDFs (Uncomment mode below to use)
    PDF_FOLDER_PATH = r"C:\path\to\your\pdfs"
    
    # 2. OUTPUT FOLDER
    OUTPUT_FOLDER = r"C:\Users\kadam\Documents\datasets\all_pages"
    
    # 3. SETTINGS
    DPI = 200               # Image quality
    
    # Page Filtering Strategy
    DEFAULT_PAGE_RANGE = "all"      # Applied to any file NOT listed below.
                                    # Examples: "all", "1-5", "10,20"

    # [OPTIONAL] Specific ranges for specific files inside the folder
    # Format: "filename.pdf": "page_range"
    PDF_SPECIFIC_RANGES = {
        "example_drawing.pdf": "1-3",
        "another_file.pdf": "50-60",
    }

    # =========================================================
    #  🚀 EXECUTION
    # =========================================================
    
    # Mode 1: Single PDF
    if os.path.exists(SINGLE_PDF_PATH):
        # Check if single file has a specific override, else use default
        fname = Path(SINGLE_PDF_PATH).name
        range_to_use = PDF_SPECIFIC_RANGES.get(fname, DEFAULT_PAGE_RANGE)
        
        pdf_to_images(SINGLE_PDF_PATH, OUTPUT_FOLDER, range_to_use, DPI)
    else:
        print(f"Note: Single PDF not found at {SINGLE_PDF_PATH}")

    # Mode 2: All PDFs in a folder (Uncomment to use)
    # if os.path.exists(PDF_FOLDER_PATH):
    #     print(f"Scanning folder: {PDF_FOLDER_PATH}")
    #     for pdf in Path(PDF_FOLDER_PATH).glob("*.pdf"):
    #         # Determine which range to use
    #         filename = pdf.name
    #         range_to_use = PDF_SPECIFIC_RANGES.get(filename, DEFAULT_PAGE_RANGE)
    #         
    #         # Run conversion
    #         pdf_to_images(str(pdf), OUTPUT_FOLDER, range_to_use, DPI)

    print("\nDone!")

if __name__ == "__main__":
    main()
