@echo off
REM Saves today's options data (put/call volume, implied volatility) for the watch list.
REM Run after 4:30 pm New York time. Results are added to data\collect_log.txt
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m atrade.collect
