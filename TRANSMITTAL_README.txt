TRANSMITTAL GUIDE - FILES TO ZIP
================================

When sending this tool to another computer, create a ZIP file containing:

1. SCRIPTS (The Code):
   - 1_prepare_images.py
   - 2_train_model.py
   - 3_extract_details.py
   - setup_project.bat
   - install_dependencies.bat

2. CONFIGURATION:
   - data.yaml
   - requirements.txt
   - README.md

3. THE TRAINED MODEL (Critical!):
   - You must include your trained model file: 'best.pt'
   - Location: runs\detect\train\weights\best.pt
   - Please include the entire 'runs' folder structure so the script finds it automatically.

HOW TO RUN ON A NEW PC:
-----------------------
1. Unzip the file.
2. Install Python (if not installed).
3. Run 'setup_project.bat' (creates necessary empty folders).
4. Run 'install_dependencies.bat' (installs YOLOv8, PyMuPDF, etc).
5. Open '3_extract_details.py' in a text editor (Notepad) to check settings.
6. Run the script:
   python 3_extract_details.py
