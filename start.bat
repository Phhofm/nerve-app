@echo off
REM Quick run for users who already have Python. Creates a venv, installs deps, launches the app.
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements-windows.txt
python app.py
pause
