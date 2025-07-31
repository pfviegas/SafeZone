@echo off
REM Child Monitor - Start Script (HTML Version) - Windows
REM Starts only the backend that serves the HTML interface

cls
echo Starting Child Monitor (HTML/Bootstrap Version)...

REM Navigate to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist ".venv" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

REM Install dependencies if necessary
echo Checking backend dependencies...
pip install -r backend\requirements.txt

REM Change directory to backend to start the server
REM cd backend

REM Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do set "IP=%%a"
set "IP=%IP: =%"

REM Start the backend (which now serves the HTML interface)
echo ========================================================
echo Starting backend server on port 8000...
echo Interface available at:
echo Local: http://localhost:8000
echo Network:  http://%IP%:8000
echo To stop, press Ctrl+C
echo ========================================================

python ./backend/main.py

deactivate
