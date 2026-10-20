@echo off
REM Downloads 25 years of daily prices for the S&P 500 stocks (plus SPY, QQQ, IWM and the VIX)
REM into the price_data folder, then offers to upload them to GitHub.
cd /d "%~dp0"
call "%LOCALAPPDATA%\academic_trading_venv\Scripts\activate.bat"
git pull --ff-only
python -m atrade.download_prices %*
if errorlevel 1 (
  echo.
  echo Some tickers failed. Running once more for just those ...
  python -m atrade.download_prices --missing-only
)
echo.
choice /M "Upload price_data to GitHub now"
if errorlevel 2 goto :end
git add price_data
git commit -m "Add price history for testing and validation"
git push
:end
pause
