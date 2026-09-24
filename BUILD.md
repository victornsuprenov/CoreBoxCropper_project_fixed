# Windows build

Production-сборка рассчитана на Windows 10/11.

Основной вариант — Python 3.14, если установленные версии OpenCV/NumPy/Pillow/PyInstaller поддерживают этот интерпретатор. Если конкретная комбинация пакетов ещё не имеет готовых wheels для Python 3.14, для сборки следует временно использовать стабильную совместимую версию Python, не меняя код приложения.

## Подготовка

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверка:

```powershell
python -c "import cv2, numpy, PIL; print(cv2.__version__); print(numpy.__version__); print(PIL.__version__)"
```

## Сборка

```bat
build.bat
```

Скрипт использует `CoreBoxCropper.spec` и собирает PyInstaller `onedir`.

Результат:

```text
dist\CoreBoxCropper\CoreBoxCropper.exe
```

EXE не требует у конечного пользователя Python, OpenCV, NumPy, Pillow или Tkinter.

## Installer

После проверки `dist\CoreBoxCropper` можно создать установщик через Inno Setup.

Рекомендуемая схема:

```text
Program Files\CoreBoxCropper\
```

Конфигурация и логи должны храниться в:

```text
%LOCALAPPDATA%\CoreBoxCropper
```

а не в `Program Files`.
