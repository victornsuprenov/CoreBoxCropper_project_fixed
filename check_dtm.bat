@echo off
chcp 65001 >nul
set "DTM=%~1"
if "%DTM%"=="" set /p "DTM=Укажите полный путь к папке DTM: "
python "%~dp0ml\validate_dataset.py" "%DTM%"
echo.
pause
