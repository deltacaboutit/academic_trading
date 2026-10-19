@echo off
REM One-time import of your options history into options_data\.
REM Put chain_history.parquet, oi_history.parquet and gex_history.csv in the "import" folder first.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m atrade.import_history --chain import\chain_history.parquet --oi import\oi_history.parquet --gex import\gex_history.csv
echo.
echo If the import finished, commit and push the options_data folder (see README).
pause
