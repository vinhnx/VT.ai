#!/usr/bin/env python3
"""
Test integration with user session to verify real-world usage.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vtai.utils.user_session_helper import get_user_session_id
from vtai.utils.litellm_integration import quick_completion
from vtai.utils.config import logger


def test_with_mock_session():
    """Test with a mock user session."""
    logger.info("Testing with mock user session...")
    
    # Mock a user session ID
    os.environ["MOCK_USER_SESSION"] = "test_user_session_123"
    
    try:
        # This should work with Ollama (local model)
        response = quick_completion(
            model="ollama/llama3.2:1b",  # Local model that should work
            messages=[
                {"role": "user", "content": "Say hello briefly"}
            ],
            user="test_user_session_123",
            max_tokens=10
        )
        
        if response:
            logger.info("✅ Mock session test successful")
            logger.info("Response: %s", response.choices[0].message.content[:100])
            return True
        else:
            logger.warning("⚠️  No response received")
            return False
            
    except Exception as e:
        logger.error("❌ Mock session test failed: %s", str(e))
        return False


def main():
    """Test the integration."""
    logger.info("Testing LiteLLM + Supabase integration with user sessions")
    logger.info("=" * 60)
    
    # Test with mock session
    success = test_with_mock_session()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ Integration test completed successfully")
        logger.info("Check your Supabase request_logs table for the logged request")
    else:
        logger.warning("⚠️  Integration test had issues")
    
    return success


if __name__ == "__main__":
    main()