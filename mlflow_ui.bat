@echo off
rem Starts the MLflow dashboard for this project at http://localhost:5000.
title MLflow - Recital
cd /d "%~dp0"
set "PY=python"
if exist "%USERPROFILE%\.venvs\contract-rag\Scripts\python.exe" set "PY=%USERPROFILE%\.venvs\contract-rag\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo Open http://localhost:5000 in your browser. Press Ctrl+C here to stop.
"%PY%" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
