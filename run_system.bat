@echo off
TITLE Prototype Alexandria - System Controller
COLOR 0A

echo ===================================================
echo      PROTOTYPE ALEXANDRIA - WINDOWS LAUNCHER
echo ===================================================
echo.
echo [1/3] Checking environment...
if not exist ".venv" (
    echo WARNING: Virtual environment not found.
    echo Please ensure you have installed dependencies: pip install -r requirements_real.txt
    pause
)

echo.
echo [2/3] Starting Backend API (Port 8000)...
start "ASI Backend API" cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Waiting 5 seconds for API to initialize...
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting Neural Interface (Port 8501)...
start "ASI Neural Interface" cmd /k "streamlit run dashboard.py"

echo.
echo [4/4] Starting Magic Folder Watchdog...
start "ASI Auto-Ingest" cmd /k "python auto_ingest.py"

echo.
echo ===================================================
echo      SYSTEM ONLINE
echo ===================================================
echo API: http://localhost:8000/docs
echo UI:  http://localhost:8501
echo.
echo Press any key to stop all services...
pause >nul

taskkill /F /IM uvicorn.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
echo System Shutdown.
