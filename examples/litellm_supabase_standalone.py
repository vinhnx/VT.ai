#!/usr/bin/env python3
"""
Standalone example demonstrating LiteLLM with Supabase logging integration.

This example shows how to:
1. Set up LiteLLM with Supabase callbacks for logging
2. Make LLM calls with automatic request/response logging  
3. Track token usage and costs per user
4. Handle different providers (OpenAI, Anthropic, Google, etc.)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vtai.utils.litellm_integration import get_llm_client, quick_completion
from vtai.utils.config import logger


def setup_environment():
    """Setup required environment variables."""
    # Supabase credentials
    if not os.environ.get("SUPABASE_URL"):
        os.environ["SUPABASE_URL"] = "your-supabase-url"
        logger.warning("SUPABASE_URL not set. Using placeholder.")
    
    if not os.environ.get("SUPABASE_ANON_KEY"):
        os.environ["SUPABASE_ANON_KEY"] = "your-supabase-anon-key"
        logger.warning("SUPABASE_ANON_KEY not set. Using placeholder.")
    
    # LLM API Keys
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. OpenAI calls will fail.")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set. Anthropic calls will fail.")


def example_openai_chat():
    """Example OpenAI chat completion with logging."""
    logger.info("Testing OpenAI chat completion...")
    
    client = get_llm_client()
    
    try:
        response = client.chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Tell me a brief fact about AI."}
            ],
            user="user_123",  # This will be logged to track usage
            temperature=0.7,
            max_tokens=100
        )
        
        logger.info("OpenAI response: %s", response.choices[0].message.content)
        logger.info("Tokens used: %s", response.usage.total_tokens if response.usage else "N/A")
        
        return response
        
    except Exception as e:
        logger.error("OpenAI chat completion failed: %s", str(e))
        return None


def example_anthropic_chat():
    """Example Anthropic Claude completion with logging."""
    logger.info("Testing Anthropic Claude completion...")
    
    try:
        response = quick_completion(
            model="claude-3-haiku-20240307",
            messages=[
                {"role": "user", "content": "What's a fascinating fact about machine learning?"}
            ],
            user="user_123",
            max_tokens=50
        )
        
        logger.info("Claude response: %s", response.choices[0].message.content)
        return response
        
    except Exception as e:
        logger.error("Anthropic completion failed: %s", str(e))
        return None


def example_google_chat():
    """Example Google Gemini completion with logging."""
    logger.info("Testing Google Gemini completion...")
    
    try:
        response = quick_completion(
            model="gemini/gemini-1.5-flash",
            messages=[
                {"role": "user", "content": "Explain quantum computing in one sentence."}
            ],
            user="user_123",
            max_tokens=30
        )
        
        logger.info("Gemini response: %s", response.choices[0].message.content)
        return response
        
    except Exception as e:
        logger.error("Google Gemini completion failed: %s", str(e))
        return None


def example_image_generation():
    """Example image generation with logging."""
    logger.info("Testing image generation...")
    
    client = get_llm_client()
    
    try:
        response = client.image_generation(
            prompt="A serene mountain landscape at sunset",
            model="dall-e-3",
            user="user_123",
            size="1024x1024",
            quality="standard"
        )
        
        logger.info("Image generated successfully: %s", response.data[0].url if response.data else "No URL")
        return response
        
    except Exception as e:
        logger.error("Image generation failed: %s", str(e))
        return None


def example_error_handling():
    """Example demonstrating error logging."""
    logger.info("Testing error handling and logging...")
    
    try:
        # This should fail and be logged as an error
        response = quick_completion(
            model="invalid-model-name",
            messages=[
                {"role": "user", "content": "This should fail"}
            ],
            user="user_123"
        )
        
    except Exception as e:
        logger.info("Expected error occurred and was logged: %s", str(e))


def demonstrate_token_tracking():
    """Demonstrate token usage tracking across multiple calls."""
    logger.info("Demonstrating token usage tracking...")
    
    user_id = "demo_user_456"
    
    # Make multiple calls to show token accumulation
    models_to_test = [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]
    
    for model in models_to_test:
        try:
            response = quick_completion(
                model=model,
                messages=[
                    {"role": "user", "content": f"Say hello using {model}"}
                ],
                user=user_id,
                max_tokens=20
            )
            
            if response and response.usage:
                logger.info(
                    "Model: %s, Tokens: %s, Cost tracked for user: %s",
                    model,
                    response.usage.total_tokens,
                    user_id
                )
                
        except Exception as e:
            logger.error("Token tracking test failed for %s: %s", model, str(e))


def main():
    """Main function demonstrating LiteLLM with Supabase logging."""
    logger.info("Starting LiteLLM + Supabase logging demonstration")
    logger.info("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Test different providers
    logger.info("Testing different LLM providers...")
    
    # OpenAI
    example_openai_chat()
    
    # Anthropic
    example_anthropic_chat()
    
    # Google
    example_google_chat()
    
    # Image generation
    example_image_generation()
    
    # Error handling
    example_error_handling()
    
    # Token tracking
    demonstrate_token_tracking()
    
    logger.info("=" * 60)
    logger.info("Demo completed! Check your Supabase request_logs table to see logged requests.")
    logger.info("Check tokens_usage table to see aggregated usage by user.")


if __name__ == "__main__":
    main()