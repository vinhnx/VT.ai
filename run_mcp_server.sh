#!/bin/bash
# Run the standalone MCP server for VT.ai

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/mcp_server" || exit 1
VENV_DIR=".venv"

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

# Always ensure requirements are installed (fix for ImportError)
uv pip install -r requirements.txt

# Environment configuration
export MCP_HOST=${MCP_HOST:-"localhost"}
export MCP_PORT=${MCP_PORT:-"9393"}
export MCP_DEFAULT_MODEL=${MCP_DEFAULT_MODEL:-"gpt-4o-mini"}

# Print the current configuration
echo "Starting MCP server with configuration:"
echo "  Host: ${MCP_HOST}"
echo "  Port: ${MCP_PORT}"
echo "  Default model: ${MCP_DEFAULT_MODEL}"

# Run the MCP server
python server.py "$@"