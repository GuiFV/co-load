@echo off
rem Runs the coload gateway. The gateway keeps its own rotating log (the
rem `log` section of the config); this script additionally captures the raw
rem console, engine start/stop command output included, to
rem %LOCALAPPDATA%\coload\coload.log.
rem Repo root is two levels up from this script.
setlocal
set "REPO=%~dp0..\.."
set "LOGDIR=%LOCALAPPDATA%\coload"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
rem Crude size cap for the console capture, which appends forever otherwise.
rem The gateway's own log rotates properly; this one just must not eat the disk.
for %%F in ("%LOGDIR%\coload.log") do if %%~zF gtr 10485760 move /y "%LOGDIR%\coload.log" "%LOGDIR%\coload.log.1" >nul
cd /d "%REPO%"
uv run coload serve >> "%LOGDIR%\coload.log" 2>&1
endlocal
