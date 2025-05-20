#!/usr/bin/env python3
"""
Simple API server for VT.ai frontend that doesn't rely on Chainlit.
"""
import asyncio
import json
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vtai.utils.llm_providers_config import DEFAULT_MODEL, MODEL_ALIAS_MAP

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    model: str = None  # Optional: allow frontend to specify model


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Simple chat endpoint that directly calls OpenAI API or other providers based on model.
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ChatResponse(
                reply="Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )

        # Model resolution logic
        requested_model = (
            req.model or os.environ.get("OPENAI_API_MODEL") or DEFAULT_MODEL
        )
        model = MODEL_ALIAS_MAP.get(requested_model, requested_model)

        # Gemini support: if model string contains 'gemini' and does not have provider, prepend 'gemini/'
        if "gemini" in model and not model.startswith("gemini/"):
            model = f"gemini/{model}"

        # Prepare the request to OpenAI or Gemini (expand as needed)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant named VT.ai.",
                },
                {"role": "user", "content": req.message},
            ],
            "temperature": 0.7,
        }

        # Choose endpoint based on model/provider (expand as needed)
        if model.startswith("gemini/"):
            endpoint = os.environ.get(
                "GEMINI_API_URL",
                "https://generativelanguage.googleapis.com/v1beta/models/"
                + model.split("/", 1)[1]
                + ":generateContent",
            )
            # Gemini expects a different payload structure; adapt as needed
            data = {"contents": [{"role": "user", "parts": [{"text": req.message}]}]}
            params = {"key": api_key}
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
            params = None

        # Make the request
        async with httpx.AsyncClient() as client:
            if params:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    params=params,
                    json=data,
                    timeout=30.0,
                )
            else:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

            if response.status_code == 200:
                response_data = response.json()
                if model.startswith("gemini/"):
                    # Gemini response parsing
                    reply = response_data["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                else:
                    reply = response_data["choices"][0]["message"]["content"]
                return ChatResponse(reply=reply)
            elif response.status_code == 429:
                return ChatResponse(
                    reply="The API key has reached its quota limit. Please try again later or use a different API key."
                )
            else:
                return ChatResponse(
                    reply=f"Error from API: {response.status_code} - {response.text}"
                )

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
