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

# Install dependencies from requirements.txt using uv
echo "Installing dependencies from requirements.txt..."
uvx pip install -r requirements.txt

# Run the VT.ai application using chainlit
echo "Starting VT.ai application..."
chainlit run vtai/app.py "$@"
