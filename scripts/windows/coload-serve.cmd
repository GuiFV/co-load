@echo off
rem Runs the coload gateway, logging to %LOCALAPPDATA%\coload\coload.log.
rem Repo root is two levels up from this script.
setlocal
set "REPO=%~dp0..\.."
set "LOGDIR=%LOCALAPPDATA%\coload"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
cd /d "%REPO%"
uv run coload serve >> "%LOGDIR%\coload.log" 2>&1
endlocal
