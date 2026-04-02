#!/bin/bash
# VT.ai All-in-One Installer and Runner
# This script installs all dependencies and runs VT.ai in one command

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
check_python() {
    print_header "Checking Python Version"
    
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        print_success "Found Python 3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        if [[ "$PYTHON_VERSION" == "3.11"* ]]; then
            PYTHON_CMD="python3"
            print_success "Found Python 3.11"
        else
            print_warning "Found Python $PYTHON_VERSION (need 3.11)"
            PYTHON_CMD="python3"
        fi
    else
        print_error "Python 3.11 not found!"
        echo "Please install Python 3.11 from https://www.python.org/downloads/"
        exit 1
    fi
    
    echo "Using: $($PYTHON_CMD --version)"
}

# Check if uv is installed, install if not
check_uv() {
    print_header "Checking uv Package Manager"
    
    if command -v uv &> /dev/null; then
        print_success "uv is already installed"
        UV_CMD="uv"
    else
        print_warning "uv not found, installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            print_success "uv installed successfully"
            UV_CMD="uv"
        else
            print_warning "uv installation failed, falling back to pip"
            UV_CMD="pip"
        fi
    fi
}

# Create virtual environment
create_venv() {
    print_header "Setting Up Virtual Environment"
    
    VENV_DIR=".venv"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            print_success "Removed old virtual environment"
        else
            print_success "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        if [ "$UV_CMD" = "uv" ]; then
            uv venv --python python3.11
        else
            $PYTHON_CMD -m venv "$VENV_DIR"
        fi
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    echo "Installing VT.ai and all dependencies (this may take a few minutes)..."
    
    if [ "$UV_CMD" = "uv" ]; then
        uv pip install -e ".[dev]"
    else
        pip install -e ".[dev]"
    fi
    
    print_success "All dependencies installed"
}

# Configure API keys
configure_api_keys() {
    print_header "Configuring API Keys"
    
    CONFIG_DIR="$HOME/.config/vtai"
    ENV_FILE="$CONFIG_DIR/.env"
    
    # Create config directory
    mkdir -p "$CONFIG_DIR"
    
    # Check if API keys already exist
    if [ -f "$ENV_FILE" ]; then
        print_warning "API configuration already exists at $ENV_FILE"
        read -p "Do you want to update it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Using existing configuration"
            return
        fi
    fi
    
    # Create or update .env file
    echo "# VT.ai API Configuration" > "$ENV_FILE"
    echo "# Created on $(date)" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"
    
    # Prompt for API keys
    echo "Enter your API keys (press Enter to skip):"
    echo ""
    
    read -p "OpenAI API Key: " OPENAI_KEY
    if [ -n "$OPENAI_KEY" ]; then
        echo "OPENAI_API_KEY='$OPENAI_KEY'" >> "$ENV_FILE"
        print_success "OpenAI API key saved"
    fi
    
    read -p "Anthropic API Key: " ANTHROPIC_KEY
    if [ -n "$ANTHROPIC_KEY" ]; then
        echo "ANTHROPIC_API_KEY='$ANTHROPIC_KEY'" >> "$ENV_FILE"
        print_success "Anthropic API key saved"
    fi
    
    read -p "Google Gemini API Key: " GEMINI_KEY
    if [ -n "$GEMINI_KEY" ]; then
        echo "GEMINI_API_KEY='$GEMINI_KEY'" >> "$ENV_FILE"
        print_success "Google Gemini API key saved"
    fi
    
    read -p "Tavily API Key (for web search): " TAVILY_KEY
    if [ -n "$TAVILY_KEY" ]; then
        echo "TAVILY_API_KEY='$TAVILY_KEY'" >> "$ENV_FILE"
        print_success "Tavily API key saved"
    fi
    
    echo ""
    print_success "API keys saved to $ENV_FILE"
    echo ""
    echo "You can always update these later by editing: $ENV_FILE"
    echo "Or set environment variables before running:"
    echo "  export OPENAI_API_KEY='your-key'"
}

# Run VT.ai
run_vtai() {
    print_header "Starting VT.ai"
    
    echo ""
    echo "Launching VT.ai..."
    echo "The application will open in your default browser at: http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    # Run chainlit
    chainlit run vtai/app -w
}

# Main installation function
main() {
    print_header "VT.ai All-in-One Installer"
    echo ""
    echo "This script will:"
    echo "  1. Check Python version (requires 3.11)"
    echo "  2. Install uv package manager (if needed)"
    echo "  3. Create a virtual environment"
    echo "  4. Install VT.ai and all dependencies"
    echo "  5. Configure API keys"
    echo "  6. Run VT.ai"
    echo ""
    
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    echo ""
    
    # Run installation steps
    check_python
    echo ""
    
    check_uv
    echo ""
    
    create_venv
    echo ""
    
    install_dependencies
    echo ""
    
    configure_api_keys
    echo ""
    
    # Ask if user wants to run now
    read -p "Do you want to run VT.ai now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_vtai
    else
        echo ""
        print_success "Installation complete!"
        echo ""
        echo "To run VT.ai later:"
        echo "  1. Activate the virtual environment:"
        echo "     source .venv/bin/activate"
        echo "  2. Run VT.ai:"
        echo "     chainlit run vtai/app"
        echo ""
        echo "Or simply run this script again to start VT.ai!"
    fi
}

# Run main function
main "$@"
