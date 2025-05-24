#!/usr/bin/env python3
"""
Test the simplified token usage system without redundant tables.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vtai.utils.supabase_logger import (
    log_request_to_supabase,
    get_user_analytics,
    get_user_monthly_usage
)
from vtai.utils.config import logger


def test_automatic_token_updates():
    """Test that token updates happen automatically via triggers."""
    logger.info("Testing automatic token updates...")
    
    test_user_id = "c02e0cf1-1efa-4b11-8068-5004eb65b822"
    
    # Get current token count
    analytics = get_user_analytics(test_user_id)
    initial_tokens = analytics["total_tokens_counter"] if analytics else 0
    
    logger.info("Current user tokens: %s", initial_tokens)
    
    # Add a new request log entry
    log_request_to_supabase(
        model="test-model",
        messages=[{"role": "user", "content": "Test automatic token update"}],
        response={"content": "Test response"},
        end_user=test_user_id,
        status="success",
        tokens_used=25,
        total_cost=0.005,
        user_profile_id=test_user_id,
        provider="test"
    )
    
    # Check if tokens were automatically updated
    updated_analytics = get_user_analytics(test_user_id)
    new_tokens = updated_analytics["total_tokens_counter"] if updated_analytics else 0
    
    logger.info("Updated user tokens: %s", new_tokens)
    
    if new_tokens == initial_tokens + 25:
        logger.info("✅ Automatic token update working correctly!")
        return True
    else:
        logger.error("❌ Automatic token update failed. Expected %s, got %s", 
                    initial_tokens + 25, new_tokens)
        return False


def test_monthly_usage_view():
    """Test the monthly usage view."""
    logger.info("Testing monthly usage view...")
    
    test_user_id = "c02e0cf1-1efa-4b11-8068-5004eb65b822"
    
    monthly_usage = get_user_monthly_usage(test_user_id)
    
    if monthly_usage:
        logger.info("✅ Monthly usage data found:")
        for month in monthly_usage:
            logger.info("  Period: %s", month.get("period_start"))
            logger.info("  Requests: %s", month.get("request_count"))
            logger.info("  Total tokens: %s", month.get("total_tokens"))
            logger.info("  Total cost: $%s", month.get("total_cost"))
            logger.info("  Model breakdown: %s", month.get("model_breakdown"))
            logger.info("  Provider breakdown: %s", month.get("provider_breakdown"))
        return True
    else:
        logger.warning("❌ No monthly usage data found")
        return False


def test_redundancy_removal():
    """Test that the system works without the redundant tokens_usage table."""
    logger.info("Testing system without redundant tables...")
    
    # This should work without the tokens_usage table
    test_user_id = "c02e0cf1-1efa-4b11-8068-5004eb65b822"
    
    # Get analytics (should work with just request_logs and user_profiles)
    analytics = get_user_analytics(test_user_id)
    
    # Get monthly usage (should work with the view)
    monthly = get_user_monthly_usage(test_user_id)
    
    if analytics and monthly:
        logger.info("✅ System working correctly without redundant tables!")
        logger.info("  User has %s total requests", analytics.get("total_requests"))
        logger.info("  Monthly data shows %s periods", len(monthly))
        return True
    else:
        logger.error("❌ System not working without redundant tables")
        return False


def main():
    """Test the simplified system."""
    logger.info("Testing Simplified Token Usage System")
    logger.info("=" * 60)
    
    results = []
    
    # Test automatic token updates
    results.append(test_automatic_token_updates())
    logger.info("-" * 40)
    
    # Test monthly usage view
    results.append(test_monthly_usage_view())
    logger.info("-" * 40)
    
    # Test redundancy removal
    results.append(test_redundancy_removal())
    
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        logger.info("✅ All simplified system tests passed (%d/%d)", passed, total)
        logger.info("🎉 System is working correctly without redundant tables!")
        logger.info("")
        logger.info("Summary:")
        logger.info("- ❌ tokens_usage table: REMOVED (redundant)")
        logger.info("- ✅ request_logs table: Contains all detailed data")
        logger.info("- ✅ user_profiles.tokens_used: Auto-updated via triggers")
        logger.info("- ✅ monthly_token_usage view: Calculates aggregations on-demand")
    else:
        logger.warning("⚠️  Some tests failed (%d/%d)", passed, total)
    
    return passed == total


if __name__ == "__main__":
    main()