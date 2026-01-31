@echo off
echo ==========================================
echo   YOLOv8 Project Installation Script
echo ==========================================
echo.
echo Installing python libraries...
echo.

pip install -r requirements.txt

echo.
echo ------------------------------------------
echo [OPTIONAL] GPU SETUP
echo ------------------------------------------
echo If this computer has an NVIDIA GPU, run the following command manualy:
echo.
echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo (Otherwise, the standard installation above works fine on CPU)
echo ------------------------------------------
echo.
echo Installation complete!
echo.
pause
