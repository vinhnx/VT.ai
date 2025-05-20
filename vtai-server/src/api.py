"""
FastAPI backend for VT.ai custom frontend.

Exposes a /api/chat endpoint that accepts a user message and returns an assistant reply.
"""

import os
import sys
from typing import Optional

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel

from vtai.utils.llm_providers_config import DEFAULT_MODEL, MODEL_ALIAS_MAP

# Add VT.ai src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Simple health check endpoint to verify the API is running.
    """
    return HealthResponse(status="ok")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Handle chat requests from the custom frontend.

    Uses litellm directly instead of depending on Chainlit context.
    """
    try:
        # Create a system message
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant named VT.ai.",
            },
            {"role": "user", "content": req.message},
        ]

        # Model resolution logic
        # Priority: request > env > default
        requested_model = (
            req.model or os.environ.get("OPENAI_API_MODEL") or DEFAULT_MODEL
        )
        # If the model is an alias, resolve it
        model = MODEL_ALIAS_MAP.get(requested_model, requested_model)

        # Gemini support: if model string contains 'gemini' and does not have provider, prepend 'gemini/'
        if "gemini" in model and not model.startswith("gemini/"):
            model = f"gemini/{model}"

        # Call the API directly
        response = await litellm.acompletion(
            model=model, messages=messages, temperature=0.7, max_tokens=1000
        )

        # Extract the reply
        reply = response.choices[0].message.content
        return ChatResponse(reply=reply)
    except AuthenticationError:
        return ChatResponse(
            reply="Error: Authentication failed. Please check your API key."
        )
    except RateLimitError:
        return ChatResponse(
            reply="The API key has reached its quota limit. Please try again later or use a different API key."
        )
    except BadRequestError as e:
        return ChatResponse(reply=f"Error in request: {str(e)}")
    except Exception as e:  # Still keep a catch-all but as last resort
        # In production, you might want to log this error but not expose details to users
        return ChatResponse(
            reply="An unexpected error occurred. Please try again later."
        )


if __name__ == "__main__":
    # When running directly, use the module name
    # This allows for proper reloading during development
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
