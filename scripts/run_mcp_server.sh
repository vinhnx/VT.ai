#!/bin/bash
# Script to run the MCP server component of VT.ai

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../mcp_server" || exit 1
VENV_DIR=".venv"

# Default port for MCP server
MCP_PORT=${MCP_PORT:-9393}

# Check if port is already in use
if lsof -i:$MCP_PORT > /dev/null; then
    echo "Port $MCP_PORT is already in use. Please specify a different port with MCP_PORT=<port>."
    exit 1
fi

# Create venv with uv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment at: $VENV_DIR using uv venv"
  uv venv "$VENV_DIR"
fi

# Activate the venv
source "$VENV_DIR/bin/activate"
echo "Using Python $(python --version) environment at: $VENV_DIR"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo "UV path: $(which uv)"

# Always install or update required packages to ensure all dependencies are present
echo "Installing required packages with uv..."
uv pip install -U -r requirements.txt

# Environment configuration
export MCP_HOST=${MCP_HOST:-"127.0.0.1"}
export MCP_PORT=$MCP_PORT
export MCP_DEFAULT_MODEL=${MCP_DEFAULT_MODEL:-"gpt-4o-mini"}
export MCP_DEV=1

# Print the current configuration
echo "Starting MCP server with configuration:"
echo "  Host: ${MCP_HOST}"
echo "  Port: ${MCP_PORT}"
echo "  Default model: ${MCP_DEFAULT_MODEL}"

# Run the MCP server
echo "Starting MCP server on port $MCP_PORT..."
python server.py "$@"