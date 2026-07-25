@echo off
rem Removes the coload autostart launcher (does not stop a running instance).
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\coload-autostart.vbs" 2>nul
echo Removed (if it was installed).
