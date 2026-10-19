@echo off
REM Starts the tool in your web browser. Close this window to stop it.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Please run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
