@echo off
title NeuroAgent - 24/7 Independent Free Cloud Deployment
color 0B
echo ===============================================================================
echo          NEUROAGENT - 24/7 FREE INDEPENDENT CLOUD DEPLOYMENT
echo ===============================================================================
echo.
echo Running automatic deployment script to Hugging Face Spaces...
echo.
python "%~dp0deploy_to_huggingface.py"
echo.
pause
