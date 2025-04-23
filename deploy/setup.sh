#!/bin/bash

# Exit if any command fails
set -e

# Optional: create and activate a virtual environment
# python3 -m venv venv
# source venv/bin/activate

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install required libraries
python3 -m pip install torch torchvision timm pandas numpy scikit-learn pillow

echo "All libraries installed successfully."

# Use these commands to run the script
# chmod +x setup.sh
# ./setup.sh
