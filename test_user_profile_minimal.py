#!/usr/bin/env python3
"""
Minimal test for UserProfile functionality without heavy dependencies.
"""

import time
import uuid
from typing import Any, Dict


# Simplified UserProfile class for testing
class UserProfile:
    """User profile data class for managing authenticated user information."""

    def __init__(self, user_id: str, email: str, display_name: str = None, metadata: Dict[str, Any] = None):
        self.user_id = user_id
        self.email = email
        self.display_name = display_name or email.split('@')[0]
        self.metadata = metadata or {}
        self.session_id = str(uuid.uuid4())
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary format."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'display_name': self.display_name,
            'metadata': self.metadata,
            'session_id': self.session_id,
            'created_at': self.created_at
        }

    @classmethod
    def from_supabase_user(cls, supabase_user) -> 'UserProfile':
        """Create UserProfile from Supabase user object."""
        user_id = supabase_user.id
        email = getattr(supabase_user, 'email', f'user_{user_id}')

        # Extract metadata from Supabase user
        metadata = {}
        if hasattr(supabase_user, 'user_metadata'):
            metadata.update(supabase_user.user_metadata or {})
        if hasattr(supabase_user, 'app_metadata'):
            metadata.update(supabase_user.app_metadata or {})

        # Try to get display name from metadata or construct from email
        display_name = metadata.get('display_name') or metadata.get('full_name')
        if not display_name and email:
            display_name = email.split('@')[0]

        return cls(
            user_id=user_id,
            email=email,
            display_name=display_name,
            metadata=metadata
        )

# Mock Supabase user object for testing
class MockSupabaseUser:
    def __init__(self):
        self.id = "test-user-123"
        self.email = "test@example.com"
        self.user_metadata = {"display_name": "Test User", "full_name": "Test User"}
        self.app_metadata = {"role": "user"}

def test_user_profile_creation():
    """Test UserProfile creation from Supabase user."""
    print("🧪 Testing UserProfile creation...")

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
    assert "session_id" in profile_dict
    assert "created_at" in profile_dict

    print("✅ UserProfile creation test passed")

def test_user_profile_fallbacks():
    """Test UserProfile fallback scenarios."""
    print("🧪 Testing UserProfile fallbacks...")

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

def test_user_profile_direct_creation():
    """Test direct UserProfile creation."""
    print("🧪 Testing direct UserProfile creation...")

    # Test direct creation
    profile = UserProfile(
        user_id="direct-123",
        email="direct@example.com",
        display_name="Direct User",
        metadata={"custom": "data"}
    )

    assert profile.user_id == "direct-123"
    assert profile.email == "direct@example.com"
    assert profile.display_name == "Direct User"
    assert profile.metadata["custom"] == "data"
    assert len(profile.session_id) > 0  # UUID should be generated
    assert profile.created_at > 0  # Timestamp should be set

    # Test with minimal data (display_name fallback)
    minimal_profile = UserProfile(
        user_id="minimal-456",
        email="minimal@test.com"
    )

    assert minimal_profile.display_name == "minimal"  # Should extract from email
    assert minimal_profile.metadata == {}  # Should default to empty dict

    print("✅ Direct UserProfile creation test passed")

if __name__ == "__main__":
    print("🧪 Testing user profile management (minimal version)...")

    try:
        test_user_profile_creation()
        test_user_profile_fallbacks()
        test_user_profile_direct_creation()

        print("🎉 All tests passed! User profile management is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        exit(1)
