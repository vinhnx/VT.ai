#!/usr/bin/env python3
"""
Standalone MCP (Model Context Protocol) Server for VT.ai.

This module provides a standalone MCP server implementation that can be run independently
from the main VT.ai application. It handles routing requests to various language models
using LiteLLM for standardization.
"""

import json
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import litellm
except ImportError:
    raise ImportError("litellm is required. Please run: uv pip install -U litellm")

# --- Config ---
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "9393"))
MCP_DEFAULT_MODEL = os.environ.get("MCP_DEFAULT_MODEL", "gpt-3.5-turbo")
MCP_MODEL_MAP = os.environ.get("MCP_MODEL_MAP", "{}")
try:
    MCP_MODEL_MAP = json.loads(MCP_MODEL_MAP)
except Exception:
    MCP_MODEL_MAP = {}


# --- Pydantic Schemas ---
class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 0.9


class CompletionResponse(BaseModel):
    content: str
    model: str
    object: str = Field(default="completion")
    id: Optional[str] = None


# --- FastAPI App ---
app = FastAPI(title="VT.ai MCP Server", version="1.0.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/completion", response_model=CompletionResponse)
async def completion(req: CompletionRequest):
    model = req.model or MCP_DEFAULT_MODEL
    model = MCP_MODEL_MAP.get(model, model)
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[msg.dict() for msg in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
        )
        content = response.choices[0].message.content if response.choices else ""
        return CompletionResponse(content=content, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "VT.ai MCP Server. See /docs for OpenAPI."}


# --- Entrypoint ---
def main():
    uvicorn.run(
        "server:app",
        host=MCP_HOST,
        port=MCP_PORT,
        reload=bool(os.environ.get("MCP_DEV", "")),
        factory=False,
    )


if __name__ == "__main__":
    main()
