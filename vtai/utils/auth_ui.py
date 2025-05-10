"""
UI components for authentication status display.

This module provides UI elements to show the current authentication status and relevant user information.
"""

import asyncio
from typing import Any, Dict, List, Optional

import chainlit as cl
from supabase import Client

from vtai.utils.config import logger


async def get_auth_actions() -> List[cl.Action]:
    """Returns a list of actions for login and sign-up."""
    return [
        cl.Action(
            name="show_login_form",
            label="🔐 Login",
            description="Login with your email and password",
            payload={},  # Empty payload as it's required but not needed for this action
        ),
        cl.Action(
            name="show_signup_form",
            label="✍️ Sign Up",
            description="Create a new account",
            payload={},  # Empty payload as it's required but not needed for this action
        ),
    ]


async def display_auth_status(
    is_authenticated: bool, user_data: Optional[Any] = None
) -> None:
    """
    Displays the current authentication status in the sidebar.

    Args:
            is_authenticated: Whether the user is authenticated.
            user_data: The user data, either from Supabase (dict) or password auth (cl.User object).
    """
    # Note: In older versions of Chainlit, we would remove previous auth messages here
    # But since the remove_elements function isn't available, we'll just send new messages
    # and let Chainlit handle any UI updates

    if is_authenticated and user_data:
        # Handle different user object structures
        if isinstance(user_data, dict):
            # Supabase user data
            user_name = user_data.get("user_metadata", {}).get(
                "full_name", user_data.get("email", "User")
            )
            user_email = user_data.get("email", "No email available")
            user_id = user_data.get("id", "Unknown ID")
            auth_provider = "Supabase"
        else:
            # Password auth user data (cl.User object)
            user_name = user_data.identifier
            user_email = "Local user"  # We don't store emails in password auth
            user_id = user_data.identifier
            auth_provider = (
                user_data.metadata.get("provider", "credentials")
                if hasattr(user_data, "metadata")
                else "Unknown"
            )

        # Display authenticated user info
        await cl.Message(
            content=f"""
## 🔐 Authenticated

**User**: {user_name}
**Email**: {user_email}
**ID**: {user_id}
**Auth Provider**: {auth_provider}

*You have full access to all VT.ai features.*
            """,
            author="Auth Status",
        ).send()
    else:
        # Display test mode info
        await cl.Message(
            content="""
## 🧪 Test Mode

You're currently using VT.ai in test mode with limited features.

Sign up or log in to access:
- Personal conversation history
- Higher usage limits
- Advanced features
- Customization options

Click the **Login** or **Sign Up** button above to authenticate.
			""",
            author="Auth Status",
        ).send()


async def show_subscription_info(
    supabase_client: Optional[Client], user_id: Optional[str]
) -> None:
    """
    Displays the current subscription information for the user.
    Placeholder for Phase 2 implementation.

    Args:
            supabase_client: The Supabase client.
            user_id: The authenticated user's ID.
    """
    if not supabase_client or not user_id:
        return

    # Check if using password authentication
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        # For password auth users, display a default subscription message
        await cl.Message(
            content="""
## 📊 Subscription Info

You're using VT.ai with a local authentication account.
All features are available in this mode.

**Plan**: Standard
**Usage**: Unlimited during development
            """,
            author="Subscription",
        ).send()
        return

    # Note: In older versions of Chainlit, we would remove previous messages here
    # But since the remove_elements function isn't available, we'll just send new messages
    # and let Chainlit handle any UI updates

    # Placeholder for Phase 2 - This will be implemented when subscription management is added
    # For now, we'll just display a basic message about subscription status

    subscription_tier = "Free Tier"  # This will come from the database in Phase 2

    await cl.Message(
        content=f"""
## 📊 Subscription Status

**Current Plan**: {subscription_tier}
**Status**: Active

*For Phase 1, all users have access to a free tier with basic functionality.*
		""",
        author="Subscription Info",
    ).send()
