@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python 3。请先安装 Python 3 并勾选 "Add Python to PATH"，然后重新运行本脚本。
  echo        下载: https://www.python.org/downloads/
  pause
  exit /b 1
)
echo 正在检查 / 安装依赖 (numpy, opencv-python, pillow, sounddevice)...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install numpy opencv-python pillow sounddevice
echo 正在安装 mediapipe (皮肤双通道遮罩用，装不上会自动回退 OpenCV)...
python -m pip install --user mediapipe
echo.
echo ========================================
echo  依赖安装完成！
echo  1) 双击 run.bat  或  运行  python gui.py
echo  2) 需要 NVIDIA 显卡 + 最新驱动
echo ========================================
pause
