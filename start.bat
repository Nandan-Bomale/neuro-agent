@echo off
echo ===================================================
echo       Starting Neuro Agent Application
echo ===================================================

echo.
echo [1/2] Starting FastAPI Backend on port 8000...
start "Neuro Agent Backend" cmd /k "cd /d "%~dp0" && uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo [2/2] Starting React Frontend...
start "Neuro Agent Frontend" cmd /k "cd /d "%~dp0frontend_react" && npm run dev"

echo.
echo All services are starting up in separate windows!
echo Once they are ready, you can access:
echo   - Unified Web UI (Recommended): http://localhost:8000/
echo   - Vite Dev Mode:                 http://localhost:5173/
echo.
echo To share online with anyone via 1-click link, run: share_online.bat
echo.
pause
