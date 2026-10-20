@echo off
REM Runs the formula checks.
cd /d "%~dp0"
call "%LOCALAPPDATA%\academic_trading_venv\Scripts\activate.bat"
python -m pytest -q tests
pause
