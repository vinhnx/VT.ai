#!/usr/bin/env bash
# VT.ai Native Installer
# Installs VT.ai with all dependencies using uv
# Supports: macOS, Linux, Windows (WSL/Git Bash)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO="vinhnx/VT.ai"
PYTHON_VERSION="3.11"
VENV_DIR=".venv"
CONFIG_DIR="${HOME}/.config/vtai"
ENV_FILE="${CONFIG_DIR}/.env"

# Ensure we're in the project root or clone if needed
if [ ! -f "pyproject.toml" ]; then
    echo -e "${BLUE}INFO:${NC} VT.ai not found in current directory"
    echo -e "${BLUE}INFO:${NC} Cloning repository..."
    git clone "https://github.com/${REPO}.git" vtai-temp
    cd vtai-temp
    trap "cd .. && rm -rf vtai-temp" EXIT
fi

# Logging functions (all output to stderr)
log_info() {
    printf '%b\n' "${BLUE}INFO:${NC} $1" >&2
}

log_success() {
    printf '%b\n' "${GREEN}✓${NC} $1" >&2
}

log_error() {
    printf '%b\n' "${RED}✗${NC} $1" >&2
}

log_warning() {
    printf '%b\n' "${YELLOW}⚠${NC} $1" >&2
}

# Check for required tools
check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v curl >/dev/null 2>&1; then
        log_error "curl is required for installation"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

# Check Python version
check_python() {
    log_info "Checking Python version (requires ${PYTHON_VERSION})"
    
    local python_cmd=""
    
    if command -v python3.11 &> /dev/null; then
        python_cmd="python3.11"
    elif command -v python3 &> /dev/null; then
        local version
        version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        if [[ "$version" == "3.11"* ]]; then
            python_cmd="python3"
        fi
    fi
    
    if [ -z "$python_cmd" ]; then
        log_error "Python ${PYTHON_VERSION} not found"
        log_info "Please install Python ${PYTHON_VERSION} from https://www.python.org/downloads/"
        exit 1
    fi
    
    log_success "Found Python $($python_cmd --version)"
    echo "$python_cmd"
}

# Install uv package manager
install_uv() {
    log_info "Checking uv package manager..."
    
    if command -v uv &> /dev/null; then
        log_success "uv is already installed"
        echo "uv"
        return
    fi
    
    log_warning "uv not found, installing..."
    
    if curl -LsSf https://astral.sh/uv/install.sh | sh &> /dev/null; then
        # Try to source the environment
        if [ -f "$HOME/.local/bin/env" ]; then
            source "$HOME/.local/bin/env" 2>/dev/null || true
        fi
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v uv &> /dev/null; then
            log_success "uv installed successfully"
            echo "uv"
            return
        fi
    fi
    
    log_warning "uv installation failed, falling back to pip"
    echo "pip"
}

# Create virtual environment
create_venv() {
    local python_cmd="$1"
    local uv_cmd="$2"
    
    log_info "Setting up virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/n) " -r
        echo >&2
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            log_success "Removed old virtual environment"
        else
            log_success "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment with Python ${PYTHON_VERSION}..."
        if [ "$uv_cmd" = "uv" ]; then
            uv venv --python "python${PYTHON_VERSION}" &> /dev/null
        else
            "$python_cmd" -m venv "$VENV_DIR"
        fi
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    local uv_cmd="$1"
    
    log_info "Installing VT.ai and all dependencies..."
    log_info "This may take a few minutes depending on your connection"
    
    if [ "$uv_cmd" = "uv" ]; then
        if uv pip install -e ".[dev]" &> /dev/null; then
            log_success "All dependencies installed"
        else
            log_error "Failed to install dependencies"
            exit 1
        fi
    else
        if pip install -e ".[dev]" &> /dev/null; then
            log_success "All dependencies installed"
        else
            log_error "Failed to install dependencies"
            exit 1
        fi
    fi
}

# Configure API keys
configure_api_keys() {
    log_info "Configuring API keys..."
    
    # Create config directory
    mkdir -p "$CONFIG_DIR"
    
    # Check if API keys already exist
    if [ -f "$ENV_FILE" ]; then
        log_warning "API configuration already exists at $ENV_FILE"
        read -p "Do you want to update it? (y/n) " -r
        echo >&2
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_success "Using existing configuration"
            return
        fi
    fi
    
    # Create new .env file
    cat > "$ENV_FILE" << EOF
# VT.ai API Configuration
# Created on $(date)

EOF
    
    # Prompt for API keys
    echo >&2
    log_info "Enter your API keys (press Enter to skip):"
    echo >&2
    
    read -p "OpenAI API Key: " OPENAI_KEY
    if [ -n "$OPENAI_KEY" ]; then
        echo "OPENAI_API_KEY='$OPENAI_KEY'" >> "$ENV_FILE"
        log_success "OpenAI API key saved"
    fi
    
    read -p "Anthropic API Key: " ANTHROPIC_KEY
    if [ -n "$ANTHROPIC_KEY" ]; then
        echo "ANTHROPIC_API_KEY='$ANTHROPIC_KEY'" >> "$ENV_FILE"
        log_success "Anthropic API key saved"
    fi
    
    read -p "Google Gemini API Key: " GEMINI_KEY
    if [ -n "$GEMINI_KEY" ]; then
        echo "GEMINI_API_KEY='$GEMINI_KEY'" >> "$ENV_FILE"
        log_success "Google Gemini API key saved"
    fi
    
    read -p "Tavily API Key (for web search): " TAVILY_KEY
    if [ -n "$TAVILY_KEY" ]; then
        echo "TAVILY_API_KEY='$TAVILY_KEY'" >> "$ENV_FILE"
        log_success "Tavily API key saved"
    fi
    
    echo >&2
    log_success "API keys saved to $ENV_FILE"
    log_info "You can always update these later by editing: $ENV_FILE"
}

# Run VT.ai
run_vtai() {
    log_info "Starting VT.ai..."
    
    echo >&2
    log_info "Launching VT.ai..."
    log_info "The application will open in your default browser at: http://localhost:8000"
    echo >&2
    log_info "Press Ctrl+C to stop the application"
    echo >&2
    
    # Run chainlit
    chainlit run vtai/app -w
}

# Show usage
show_usage() {
    cat <<'USAGE'
VT.ai Native Installer

Usage: ./install_and_run.sh [options]

Options:
    --no-run          Install but don't run VT.ai
    --no-api-config   Skip API key configuration
    -h, --help        Show this help message

Examples:
    ./install_and_run.sh              # Full installation and run
    ./install_and_run.sh --no-run     # Install only
    ./install_and_run.sh --no-api-config  # Skip API prompts

Environment variables:
    OPENAI_API_KEY       Set OpenAI API key
    ANTHROPIC_API_KEY    Set Anthropic API key
    GEMINI_API_KEY       Set Google Gemini API key
    TAVILY_API_KEY       Set Tavily API key

USAGE
}

# Main installation flow
main() {
    local run_after_install=1
    local configure_api=1
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-run)
                run_after_install=0
                shift
                ;;
            --no-api-config)
                configure_api=0
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo >&2
    log_info "VT.ai Native Installer"
    echo >&2
    log_info "This script will:"
    log_info "  1. Check Python version (requires ${PYTHON_VERSION})"
    log_info "  2. Install uv package manager (if needed)"
    log_info "  3. Create a virtual environment"
    log_info "  4. Install VT.ai and all dependencies"
    if [ $configure_api -eq 1 ]; then
        log_info "  5. Configure API keys"
    fi
    if [ $run_after_install -eq 1 ]; then
        log_info "  $(($configure_api ? 6 : 5)). Run VT.ai"
    fi
    echo >&2
    
    read -p "Continue? (y/n) " -r
    echo >&2
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled."
        exit 0
    fi
    
    echo >&2
    
    # Run installation steps
    check_requirements
    echo >&2
    
    local python_cmd
    python_cmd=$(check_python)
    echo >&2
    
    local uv_cmd
    uv_cmd=$(install_uv)
    echo >&2
    
    create_venv "$python_cmd" "$uv_cmd"
    echo >&2
    
    install_dependencies "$uv_cmd"
    echo >&2
    
    if [ $configure_api -eq 1 ]; then
        configure_api_keys
        echo >&2
    fi
    
    # Ask if user wants to run now
    if [ $run_after_install -eq 1 ]; then
        read -p "Do you want to run VT.ai now? (y/n) " -r
        echo >&2
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_vtai
        else
            echo >&2
            log_success "Installation complete!"
            echo >&2
            log_info "To run VT.ai later:"
            log_info "  1. Activate the virtual environment:"
            log_info "     source ${VENV_DIR}/bin/activate"
            log_info "  2. Run VT.ai:"
            log_info "     chainlit run vtai/app"
            echo >&2
            log_info "Or simply run this script again to start VT.ai!"
        fi
    else
        echo >&2
        log_success "Installation complete!"
        echo >&2
        log_info "To run VT.ai:"
        log_info "  1. Activate the virtual environment:"
        log_info "     source ${VENV_DIR}/bin/activate"
        log_info "  2. Run VT.ai:"
        log_info "     chainlit run vtai/app"
        echo >&2
    fi
}

# Run main function
main "$@"
