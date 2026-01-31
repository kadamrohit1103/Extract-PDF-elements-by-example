@echo off
echo ==========================================
echo   YOLOv8 Project Setup Script
echo ==========================================
echo.
echo Creating directory structure...
echo.

if not exist "train\images" mkdir "train\images"
if not exist "train\labels" mkdir "train\labels"
if not exist "val\images" mkdir "val\images"
if not exist "val\labels" mkdir "val\labels"
if not exist "all_pages" mkdir "all_pages"
if not exist "runs" mkdir "runs"
if not exist "figures" mkdir "figures"

echo.
echo [OK] Directories created successfully!
echo.
echo Next steps:
echo 1. Put PDF files in the project folder
echo 2. Run 'python 1_prepare_images.py'
echo.
pause
