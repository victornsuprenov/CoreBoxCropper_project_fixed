@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found.
  echo Create it with: py -3.14 -m venv .venv
  exit /b 1
)

call .venv\Scripts\activate
python -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%

python -m PyInstaller --noconfirm --clean CoreBoxCropper.spec
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete:
echo dist\CoreBoxCropper\CoreBoxCropper.exe
