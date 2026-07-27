@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launch.py
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python launch.py
  ) else (
    echo Python 3 is required. Install it from https://www.python.org/downloads/
    pause
  )
)
