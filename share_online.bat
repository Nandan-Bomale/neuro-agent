@echo off
title NeuroAgent - Instant 1-Click Public Deployment
color 0A
echo ===============================================================================
echo                NEUROAGENT - 1-CLICK FREE ONLINE DEPLOYMENT
echo ===============================================================================
echo.
echo [1/3] Checking Backend status...

netstat -ano | findstr :8000 | findstr LISTENING >nul
if %ERRORLEVEL% equ 0 (
    echo [*] Backend is already running on port 8000!
) else (
    echo [*] Starting NeuroAgent Backend server on port 8000...
    start "NeuroAgent Server" /min cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
    timeout /t 5 /nobreak >nul
)

echo.
echo [2/3] Checking Cloudflare Tunnel...
if not exist "%~dp0tools\cloudflared.exe" (
    echo [*] Downloading standalone cloudflared binary...
    if not exist "%~dp0tools" mkdir "%~dp0tools"
    curl.exe -L -o "%~dp0tools\cloudflared.exe" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)

echo.
echo [3/3] Generating Public HTTPS 1-Click Link...
echo ===============================================================================
echo  Your web application is being connected to the global Cloudflare network.
echo  LOOK FOR THE LINK BELOW ENDING IN .trycloudflare.com
echo  Anyone in the world can click that link to access your live system!
echo ===============================================================================
echo.
"%~dp0tools\cloudflared.exe" tunnel --url http://localhost:8000

pause
