#!/usr/bin/env python3
"""
Test script to verify user profile management integration.

This script tests the user profile functionality without running the full server.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock Supabase user object for testing
class MockSupabaseUser:
    def __init__(self):
        self.id = "test-user-123"
        self.email = "test@example.com"
        self.user_metadata = {"display_name": "Test User", "full_name": "Test User"}
        self.app_metadata = {"role": "user"}


def test_user_profile_creation():
    """Test UserProfile creation from Supabase user."""
    from vtai.app import UserProfile

    # Create mock user
    mock_user = MockSupabaseUser()

    # Create user profile
    profile = UserProfile.from_supabase_user(mock_user)

    # Verify profile data
    assert profile.user_id == "test-user-123"
    assert profile.email == "test@example.com"
    assert profile.display_name == "Test User"
    assert "display_name" in profile.metadata
    assert "role" in profile.metadata

    # Test dictionary conversion
    profile_dict = profile.to_dict()
    assert profile_dict["user_id"] == "test-user-123"
    assert profile_dict["email"] == "test@example.com"
    assert profile_dict["display_name"] == "Test User"

    print("✅ UserProfile creation test passed")


def test_user_profile_fallbacks():
    """Test UserProfile fallback scenarios."""
    from vtai.app import UserProfile

    # Test with minimal user data
    class MinimalUser:
        def __init__(self):
            self.id = "minimal-user"
            # No email attribute
            # No metadata

    minimal_user = MinimalUser()
    profile = UserProfile.from_supabase_user(minimal_user)

    # Should fallback to user_id for email
    assert profile.email == "user_minimal-user"
    assert profile.display_name == "user_minimal-user"  # Should fallback to email split

    print("✅ UserProfile fallback test passed")


def test_global_profile_management():
    """Test global user profile state management."""
    from vtai.app import UserProfile, get_current_user_profile, set_current_user_profile

    # Initially no profile
    assert get_current_user_profile() is None

    # Set profile
    mock_user = MockSupabaseUser()
    profile = UserProfile.from_supabase_user(mock_user)
    set_current_user_profile(profile)

    # Retrieve profile
    retrieved_profile = get_current_user_profile()
    assert retrieved_profile is not None
    assert retrieved_profile.user_id == "test-user-123"
    assert retrieved_profile.email == "test@example.com"

    # Clear profile
    set_current_user_profile(None)
    assert get_current_user_profile() is None

    print("✅ Global profile management test passed")


if __name__ == "__main__":
    print("🧪 Testing user profile management...")

    try:
        test_user_profile_creation()
        test_user_profile_fallbacks()
        test_global_profile_management()

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
        sys.exit(1)
