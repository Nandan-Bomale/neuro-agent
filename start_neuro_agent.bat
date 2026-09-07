@echo off
echo ==================================================
echo Neuro Agent - Startup Script
echo ==================================================
echo.
echo [1/2] Starting FastAPI Backend on port 8000...
start "Neuro Agent Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo [2/2] Starting Streamlit Frontend...
start "Neuro Agent Frontend" cmd /k "python -m streamlit run frontend/app.py"

echo.
echo Both servers have been launched in separate windows!
echo - Backend API: http://localhost:8000/docs
echo - Frontend UI: http://localhost:8501
echo.
pause
