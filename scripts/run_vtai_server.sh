#!/bin/bash
# Run the standalone MCP server for VT.ai

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../mcp_server" || exit 1
VENV_DIR=".venv"

# Create venv if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Activate the venv
source "$VENV_DIR/bin/activate"
echo "Using Python $(python --version) environment at: $VENV_DIR"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# Ensure 'uv' is installed in the venv
if [[ ! -x "$VENV_DIR/bin/uv" ]]; then
  echo "Installing 'uv' in the venv..."
  "$VENV_DIR/bin/pip" install -U uv
fi

# Always ensure requirements are installed (fix for ImportError)
"$VENV_DIR/bin/uv" pip install -r requirements.txt

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
"$VENV_DIR/bin/python" server.py "$@"