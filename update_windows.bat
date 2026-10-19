@echo off
REM Gets the newest code and options data from GitHub.
cd /d "%~dp0"
git pull --ff-only
pause
