"""
server.py - FastAPI/Chainlit server setup for VT.ai

This module initializes the FastAPI app, mounts static files, and adds authentication middleware.
"""

import os

import chainlit.server
from fastapi.staticfiles import StaticFiles

from vtai.app import auth_middleware, logger, supabase_client

# Expose the FastAPI app
app = chainlit.server.app

# Mount the public directory at /public for static assets (not at /)
public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
if os.path.isdir(public_dir):
    app.mount("/public", StaticFiles(directory=public_dir), name="public")
    logger.info(f"Mounted public static files at /public from {public_dir}")

# Add authentication middleware (always enabled, no VT_ENABLE_AUTH check)
if supabase_client:
    app.middleware("http")(auth_middleware)
    logger.info("Authentication middleware added to Chainlit FastAPI app.")
else:
    logger.info("Authentication middleware not enabled or Supabase not configured.")
