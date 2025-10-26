#!/bin/bash

# MLOps Pipeline Setup Script
# This script sets up the development environment for the MLOps pipeline

set -e

echo "🚀 Setting up MLOps Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_status "Python version: $PYTHON_VERSION ✓"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating project directories..."
mkdir -p logs models data

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    print_status "Setting up pre-commit hooks..."
    pre-commit install
else
    print_warning "pre-commit not found. Install it with: pip install pre-commit"
fi

# Train initial model
print_status "Training initial model..."
python src/train.py --model logistic

# Run tests
print_status "Running tests..."
python -m pytest tests/ -v

print_status "✅ Setup complete!"
print_status ""
print_status "To start the API server:"
print_status "  source venv/bin/activate"
print_status "  uvicorn src.api:app --reload"
print_status ""
print_status "To run with Docker:"
print_status "  docker-compose up --build"
print_status ""
print_status "API will be available at: http://localhost:8000"
print_status "API documentation: http://localhost:8000/docs"
