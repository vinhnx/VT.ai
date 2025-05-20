#!/bin/bash
# Run the VT.ai FastAPI backend for the custom frontend
cd "$(dirname "$0")/../vtai-server"
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
