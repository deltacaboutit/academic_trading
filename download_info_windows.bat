@echo off
REM Downloads each stock's exchange, sector and shares outstanding into price_data\info.csv,
REM then offers to upload it to GitHub. Takes about 2 minutes.
cd /d "%~dp0"
call "%LOCALAPPDATA%\academic_trading_venv\Scripts\activate.bat"
git pull --ff-only
python -m atrade.download_prices --info
echo.
choice /M "Upload price_data\info.csv to GitHub now"
if errorlevel 2 goto :end
git add price_data\info.csv
git commit -m "Add company info (exchange, sector, shares outstanding)"
git push
:end
pause
