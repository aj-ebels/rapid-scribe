@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found. Create one with: python -m venv .venv
    echo Then install deps: .venv\Scripts\python.exe -m pip install -r requirements.txt
)
python main.py
if errorlevel 1 pause
