#!/bin/bash
# Build and run the Rust version of VT.ai

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building rust-vtai...${NC}"

# Navigate to the project directory
cd "$(dirname "$0")"

# Check for and remove obsolete src/app.rs if it exists
if [ -f "src/app.rs" ]; then
    echo -e "${BLUE}Removing obsolete src/app.rs file...${NC}"
    rm src/app.rs
fi

# Check if any API keys were provided
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${RED}Warning: No API keys detected in environment variables.${NC}"
    echo -e "You can set them with:"
    echo -e "  export OPENAI_API_KEY=sk-..."
    echo -e "  export ANTHROPIC_API_KEY=sk-..."
    echo -e "  export DEEPSEEK_API_KEY=sk-..."
    echo -e "Or you can provide them when running the application with --api-key parameter."
fi

# Build the project in release mode
echo -e "${BLUE}Running cargo build --release${NC}"
cargo build --release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"

    # Parse command line arguments for API keys
    while [[ $# -gt 0 ]]; do
        case $1 in
            --openai-key)
                export OPENAI_API_KEY="$2"
                shift 2
                ;;
            --anthropic-key)
                export ANTHROPIC_API_KEY="$2"
                shift 2
                ;;
            --deepseek-key)
                export DEEPSEEK_API_KEY="$2"
                shift 2
                ;;
            *)
                # Collect all other args to pass to the binary
                ARGS+=" $1"
                shift
                ;;
        esac
    done

    # Ask for API keys if not set in env or via args
    if [ -z "$OPENAI_API_KEY" ]; then
        read -p "Enter your OpenAI API key (sk-...): " OPENAI_API_KEY
        export OPENAI_API_KEY
    fi
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        read -p "Enter your Anthropic API key (sk-...): " ANTHROPIC_API_KEY
        export ANTHROPIC_API_KEY
    fi
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        read -p "Enter your DeepSeek API key (sk-...): " DEEPSEEK_API_KEY
        export DEEPSEEK_API_KEY
    fi

    echo -e "${BLUE}Running rust-vtai $ARGS${NC}"
    echo -e "${BLUE}--------------------------------------------------${NC}"

    # Run the binary with any arguments passed to this script
    ./target/release/vtai $ARGS
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi