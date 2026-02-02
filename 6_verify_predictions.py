"""
Step 6: Verify and Correct Predictions
A simple tool to quickly review model predictions and build training data.
"""

import os
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# =========================================================
#  🔧 CONFIGURATION - EDIT THIS SECTION
# =========================================================

IMAGES_FOLDER = r"C:\Users\kadam\Documents\datasets\figures"
PREDICTIONS_FILE = r"C:\Users\kadam\Documents\datasets\component_predictions.xlsx"
LABELS_FILE = r"C:\Users\kadam\Documents\datasets\labels.csv"
COMPONENTS_FILE = r"C:\Users\kadam\Documents\datasets\components.txt"

# =========================================================
#  Verification GUI
# =========================================================

class VerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Component Verification Tool")
        self.root.geometry("1000x700")
        
        # Load data
        self.load_data()
        self.current_idx = 0
        
        # Create UI
        self.create_widgets()
        self.show_current_image()
    
    def load_data(self):
        """Load predictions and components."""
        # Load predictions
        if os.path.exists(PREDICTIONS_FILE):
            self.predictions = pd.read_excel(PREDICTIONS_FILE)
        else:
            messagebox.showerror("Error", f"Predictions file not found: {PREDICTIONS_FILE}")
            self.root.destroy()
            return
        
        # Load components
        self.components = []
        if os.path.exists(COMPONENTS_FILE):
            with open(COMPONENTS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.components.append(line)
        
        # Track verified/corrected labels
        self.verified = {}
    
    def create_widgets(self):
        """Create the UI."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Image display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(left_frame, text="Loading...")
        self.image_label.pack(pady=10)
        
        self.info_label = ttk.Label(left_frame, text="", font=('Arial', 10))
        self.info_label.pack(pady=5)
        
        # Right: Controls
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Prediction display
        ttk.Label(right_frame, text="Model Prediction:", font=('Arial', 11, 'bold')).pack(anchor='w')
        self.pred_label = ttk.Label(right_frame, text="", wraplength=280)
        self.pred_label.pack(anchor='w', pady=5)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Component checkboxes
        ttk.Label(right_frame, text="Correct Labels:", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        # Scrollable checkbox area
        canvas = tk.Canvas(right_frame, height=300)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.checkbox_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        canvas.pack(side=tk.LEFT, fill='both', expand=True)
        canvas.create_window((0, 0), window=self.checkbox_frame, anchor='nw')
        
        self.checkbox_vars = {}
        for comp in self.components:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(self.checkbox_frame, text=comp, variable=var)
            cb.pack(anchor='w')
            self.checkbox_vars[comp] = var
        
        self.checkbox_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="✓ Accept Prediction", command=self.accept_prediction).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="✓ Save Corrections", command=self.save_corrections).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="Skip", command=self.skip_image).pack(fill='x', pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill='x', pady=10)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image).pack(side=tk.RIGHT)
        
        # Progress
        self.progress_label = ttk.Label(right_frame, text="")
        self.progress_label.pack(pady=10)
        
        # Save all button
        ttk.Button(right_frame, text="💾 Save All to labels.csv", command=self.save_all).pack(fill='x', pady=10)
    
    def show_current_image(self):
        """Display current image and prediction."""
        if self.current_idx >= len(self.predictions):
            self.current_idx = len(self.predictions) - 1
        if self.current_idx < 0:
            self.current_idx = 0
        
        row = self.predictions.iloc[self.current_idx]
        image_name = row['image']
        components = str(row['components']) if pd.notna(row['components']) else ""
        
        # Load and display image
        image_path = os.path.join(IMAGES_FOLDER, image_name)
        try:
            img = Image.open(image_path)
            img.thumbnail((500, 500))
            self.photo = tk.PhotoImage(file=image_path) if image_path.endswith('.png') else None
            
            # For other formats, we need PIL ImageTk
            from PIL import ImageTk
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.photo)
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {e}")
        
        # Update info
        self.info_label.configure(text=f"File: {image_name}")
        self.pred_label.configure(text=components if components else "(none detected)")
        
        # Update checkboxes
        for comp, var in self.checkbox_vars.items():
            var.set(comp in components)
        
        # Update progress
        verified_count = len(self.verified)
        self.progress_label.configure(
            text=f"Image {self.current_idx + 1} / {len(self.predictions)} | Verified: {verified_count}"
        )
    
    def accept_prediction(self):
        """Accept the model's prediction as correct."""
        row = self.predictions.iloc[self.current_idx]
        components = str(row['components']) if pd.notna(row['components']) else ""
        self.verified[row['image']] = components.split(', ') if components else []
        self.next_image()
    
    def save_corrections(self):
        """Save the user's corrections."""
        row = self.predictions.iloc[self.current_idx]
        selected = [comp for comp, var in self.checkbox_vars.items() if var.get()]
        self.verified[row['image']] = selected
        self.next_image()
    
    def skip_image(self):
        """Skip without saving."""
        self.next_image()
    
    def next_image(self):
        """Move to next image."""
        self.current_idx += 1
        if self.current_idx >= len(self.predictions):
            messagebox.showinfo("Complete", "You've reached the end!")
            self.current_idx = len(self.predictions) - 1
        self.show_current_image()
    
    def prev_image(self):
        """Move to previous image."""
        self.current_idx -= 1
        self.show_current_image()
    
    def save_all(self):
        """Save all verified labels to labels.csv."""
        if not self.verified:
            messagebox.showwarning("Warning", "No verified labels to save!")
            return
        
        # Append to existing labels file
        with open(LABELS_FILE, 'a') as f:
            for image_name, components in self.verified.items():
                if components:
                    line = f"{image_name},{','.join(components)}\n"
                    f.write(line)
        
        messagebox.showinfo("Saved", f"Saved {len(self.verified)} verified labels to {LABELS_FILE}")
        self.verified.clear()

# =========================================================
#  Main
# =========================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Component Verification Tool                                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  - View each image and its prediction                        ║
    ║  - Click "Accept" if correct, or check boxes and "Save"      ║
    ║  - Click "Save All" when done to update labels.csv           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    root = tk.Tk()
    app = VerificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
