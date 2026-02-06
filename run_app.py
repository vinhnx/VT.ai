#!/usr/bin/env python3
"""
Chainlit-compatible wrapper script for VT.ai application.
This script properly exposes the VT.ai callbacks to Chainlit.
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any

# Add the project root to the Python path so imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set the current working directory to the project root
os.chdir(project_root)

# Import all the required callbacks from the main app
from vtai.app import (
    build_chat_profile,
    start_chat,
    on_message
)

def main():
    """
    Main entry point for the VT.ai application.
    This function can be used to run the application programmatically.
    """
    print("Starting VT.ai application...")
    
    # Initialize the application
    try:
        # Import chainlit here to avoid issues when running as standalone
        import chainlit as cl
        
        # Expose the callbacks to Chainlit
        globals()['build_chat_profile'] = build_chat_profile
        globals()['start_chat'] = start_chat
        globals()['on_message'] = on_message
        
        print("VT.ai application initialized successfully!")
        print("Access the interface at: http://localhost:8000")
        
        # If this script is run directly, start the Chainlit server
        if __name__ == "__main__":
            # This will be handled by the chainlit command
            pass
            
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing VT.ai application: {e}")
        sys.exit(1)

# Expose the callbacks to Chainlit
__all__ = ['build_chat_profile', 'start_chat', 'on_message']

# Call main if this script is run directly
if __name__ == "__main__":
    main()