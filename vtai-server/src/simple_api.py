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


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Simple chat endpoint that directly calls OpenAI API
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ChatResponse(
                reply="Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )

        # Prepare the request to OpenAI
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        data = {
            "model": "gpt-4o-nano",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant named VT.ai.",
                },
                {"role": "user", "content": req.message},
            ],
            "temperature": 0.7,
        }

        # Make the request to OpenAI
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0,
            )

            if response.status_code == 200:
                response_data = response.json()
                reply = response_data["choices"][0]["message"]["content"]
                return ChatResponse(reply=reply)
            elif response.status_code == 429:
                # Handle quota exceeded error more gracefully
                return ChatResponse(
                    reply="The OpenAI API key has reached its quota limit. Please try again later or use a different API key."
                )
            else:
                return ChatResponse(
                    reply=f"Error from OpenAI API: {response.status_code} - {response.text}"
                )

    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
