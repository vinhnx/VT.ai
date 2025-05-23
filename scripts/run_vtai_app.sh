#!/bin/bash

# Script to run the VT.ai application

# Set the working directory to the VT.ai project root directory
cd "$(dirname "$0")/.." || { echo "Error: Cannot find VT.ai root directory"; exit 1; }

# Load environment variables if .env file exists
if [ -f .env ]; then
  echo "Loading environment variables from .env"
  set -a
  source .env
  set +a
fi

# Only check dependencies if --check-deps flag is provided
if [[ "$*" == *"--check-deps"* ]]; then
  echo "Checking dependencies from requirements.txt..."
  uvx pip install -r requirements.txt
else
  echo "Skipping dependency check. Use --check-deps flag to verify all dependencies."
fi

# Activate the virtual environment before running the app
if [ -d ".venv" ]; then
  echo "Activating .venv..."
  source .venv/bin/activate
fi

# Run the VT.ai application using uvicorn with the new main_app
echo "Starting VT.ai application with Uvicorn..."
export VT_FAST_START=0
export VT_SKIP_MODEL_PRICES=1
uvicorn vtai.server:main_app --host 0.0.0.0 --port 8000 --reload "$@"
