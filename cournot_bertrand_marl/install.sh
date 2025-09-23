#!/bin/bash

# Installation script for Cournot-Bertrand MARL project

echo "=== Installing Cournot-Bertrand MARL Project ==="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo ""
echo "=== Installation completed successfully! ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demo, execute:"
echo "  python demo.py"
echo ""
echo "To run the Jupyter notebook, execute:"
echo "  jupyter notebook notebooks/01_cournot_basics.ipynb"
echo ""
echo "To run tests, execute:"
echo "  pytest tests/"
