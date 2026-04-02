#!/usr/bin/env python3
"""
VT.ai Command Line Interface

A minimal CLI for VT.ai application with version and help support.
"""

import sys


def get_version():
    """Get the version string without triggering full app initialization."""
    try:
        from importlib.metadata import version
        return version("vtai")
    except Exception:
        return "0.7.7"


def main():
    """Main entry point for the VT.ai CLI."""
    # Check for --version or -v first before ANY imports
    if len(sys.argv) == 2 and sys.argv[1] in ["--version", "-v"]:
        print(f"vtai {get_version()}")
        sys.exit(0)
    
    # Check for --help or -h
    if len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]:
        print("""VT.ai - Minimal multimodal AI chat app with dynamic conversation routing

Usage: vtai [OPTIONS]

Options:
  -v, --version     Show the version number and exit
  -h, --help        Show this help message and exit

Examples:
  vtai                    Start the VT.ai application
  vtai --version          Show the version number
  vtai --help             Show this help message

For more information, visit: https://github.com/vinhnx/VT.ai
""")
        sys.exit(0)

    # For all other cases, run the application via chainlit
    # We invoke chainlit directly to avoid importing the app module
    import os
    
    # Set environment variable to indicate we're running in CLI mode
    os.environ.setdefault("CHAINLIT_RUN_WITHOUT_WATCH", "1")
    
    # Find the app.py file
    import os.path
    from pathlib import Path
    
    # Get the directory where this CLI module is located
    cli_dir = Path(__file__).parent
    app_path = cli_dir / "app.py"
    
    try:
        import chainlit.cli
        
        # Set up arguments to run the app with chainlit
        # Filter out any chainlit-specific args that might interfere
        sys.argv = [
            "chainlit",
            "run",
            str(app_path),
            "--headless",
        ]
        
        # Run the chainlit application
        chainlit.cli.cli()
    except ImportError:
        print("Error: chainlit is not installed or not available.")
        print("Please install the package with: pip install vtai")
        sys.exit(1)
    except SystemExit as e:
        # Re-raise SystemExit to allow proper exit
        raise
    except Exception as e:
        print(f"Error running VT.ai: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Handle version/help before anything else
    if len(sys.argv) == 2 and sys.argv[1] in ["--version", "-v"]:
        try:
            from importlib.metadata import version
            print(f"vtai {version('vtai')}")
        except Exception:
            print("vtai 0.7.7")
        sys.exit(0)
    
    if len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]:
        print("""VT.ai - Minimal multimodal AI chat app with dynamic conversation routing

Usage: vtai [OPTIONS]

Options:
  -v, --version     Show the version number and exit
  -h, --help        Show this help message and exit

Examples:
  vtai                    Start the VT.ai application
  vtai --version          Show the version number
  vtai --help             Show this help message

For more information, visit: https://github.com/vinhnx/VT.ai
""")
        sys.exit(0)
    
    main()
