@echo off
REM Runs the formula checks.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m pytest -q tests
pause
