@echo off
REM Starts the tool in your web browser. Close this window to stop it.
cd /d "%~dp0"
REM The environment lives outside OneDrive (avoids sync locks and the 260-char path limit).
set VENV=%LOCALAPPDATA%\academic_trading_venv
if not exist "%VENV%\Scripts\activate.bat" (
  echo Please run setup_windows.bat first.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"
streamlit run app.py
