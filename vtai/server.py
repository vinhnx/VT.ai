"""
server.py - FastAPI/Chainlit server setup for VT.ai

This module initializes the FastAPI app, mounts static files, and adds authentication middleware.
"""

import os

import chainlit.server
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from vtai.app import auth_middleware, logger, supabase_client

# Create a new main FastAPI application
main_app = FastAPI()

# Register authentication middleware as the very first middleware for the main_app
main_app.middleware("http")(auth_middleware)
logger.info("[SERVER] Authentication middleware registered to main_app.")

# Mount the public directory at /public for static assets
public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
if os.path.isdir(public_dir):
    main_app.mount("/public", StaticFiles(directory=public_dir), name="public")
    logger.info(f"Mounted public static files at /public from {public_dir} on main_app")


@main_app.api_route("/mcp", methods=["POST"])
async def proxy_mcp(request: Request) -> Response:
    """
    Proxy /mcp requests to the MCP server at http://localhost:9393/completion.
    This is required for compatibility with Chainlit's frontend, which expects /mcp on port 8000.
    """
    try:
        # Forward all headers except host
        headers = dict(request.headers)
        headers.pop("host", None)
        # Read the raw body
        body = await request.body()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:9393/completion",
                headers=headers,
                content=body,
                timeout=30.0,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() != "content-encoding"
                },
            )
    except Exception as e:
        return JSONResponse(
            status_code=502, content={"error": f"MCP proxy error: {str(e)}"}
        )


# Get the Chainlit app
cl_app = chainlit.server.app

# Mount the Chainlit app to the main_app at /
# All Chainlit's internal routes (including /, /ws, /static-files, etc.) will be under /
main_app.mount("/", cl_app)
logger.info("[SERVER] Chainlit app (cl_app) mounted at / on main_app.")

# Note: The original `app = chainlit.server.app` is no longer the primary app.
# `main_app` is now the primary app to be run with Uvicorn.
