"""
server.py - FastAPI/Chainlit server setup for VT.ai

This module initializes the FastAPI app, mounts static files, and adds authentication (if any).
"""

import os

import chainlit.server
import httpx
from app import logger, supabase_client
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Create a new main FastAPI application
main_app = FastAPI()

# Add CORS middleware first to allow cross-origin requests from the frontend
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
logger.info(
    "[SERVER] CORS middleware registered, allowing origin http://localhost:3000."
)

# Mount the public directory at /public for static assets
public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
if os.path.isdir(public_dir):
    main_app.mount("/public", StaticFiles(directory=public_dir), name="public")
    logger.info(f"Mounted public static files at /public from {public_dir} on main_app")


# Get the Chainlit app
cl_app = chainlit.server.app

# Mount the Chainlit app to the main_app at /
# All Chainlit's internal routes (including /, /ws, /static-files, etc.) will be under /
main_app.mount("/", cl_app)
logger.info("[SERVER] Chainlit app (cl_app) mounted at / on main_app.")

# Note: The original `app = chainlit.server.app` is no longer the primary app.
# `main_app` is now the primary app to be run with Uvicorn.
