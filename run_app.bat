@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        where python >nul 2>nul
        if errorlevel 1 goto python_missing
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 goto setup_failed
)

".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 goto python_version

if not exist ".venv\Scripts\streamlit.exe" (
    echo First run: installing app dependencies...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_failed
)

echo Starting AI Text Detector in your browser...
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" echo The app exited with error code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:python_missing
echo Python 3.9 or newer was not found. Install Python and try again.
pause
exit /b 1

:python_version
echo Python 3.9 or newer is required. Install a supported version and try again.
pause
exit /b 1

:setup_failed
echo Setup failed. Check your Python installation and internet connection, then try again.
pause
exit /b 1
