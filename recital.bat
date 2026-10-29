@echo off
rem Starts the Recital web app. Open http://localhost:8000 once it says it's running.
rem Ollama must be running for answers (open the Ollama app if it isn't).
title Recital
cd /d "%~dp0"
call :find_python
"%PY%" app.py
goto :eof

:find_python
rem Prefer a project .venv, then a venv under %USERPROFILE%\.venvs, then python on PATH.
set "PY=python"
if exist "%USERPROFILE%\.venvs\contract-rag\Scripts\python.exe" set "PY=%USERPROFILE%\.venvs\contract-rag\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
goto :eof
