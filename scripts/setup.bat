@echo off
REM MLOps Pipeline Setup Script for Windows
REM This script sets up the development environment for the MLOps pipeline

echo 🚀 Setting up MLOps Pipeline...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed. Please install Python 3.9+ and try again.
    exit /b 1
)

echo [INFO] Python is installed ✓

REM Create virtual environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
) else (
    echo [WARNING] Virtual environment already exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo [INFO] Creating project directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data" mkdir data

REM Train initial model
echo [INFO] Training initial model...
python src/train.py --model logistic

REM Run tests
echo [INFO] Running tests...
python -m pytest tests/test_api.py -v

echo [INFO] Setup complete!
echo.
echo [INFO] To start the API server:
echo   venv\Scripts\activate
echo   uvicorn src.api:app --reload
echo.
echo [INFO] To run with Docker:
echo   docker-compose up --build
echo.
echo [INFO] API will be available at: http://localhost:8000
echo [INFO] API documentation: http://localhost:8000/docs

pause
