"""
Core API server for VT.ai.

This module sets up the FastAPI application, including routes for chat,
subscription management, and other core functionalities as outlined in the
VT.ai monetization plan. It integrates with Supabase for user authentication
and data storage, and Stripe for payment processing.
"""

import os
from typing import List, Optional

import dotenv
import stripe
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from gotrue.types import User as GoTrueUser
from litellm import completion
from pydantic import BaseModel
from supabase import Client, create_client

# Attempt to import the configured logger
try:
    from vtai.utils.config import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="VT.ai API",
    description="API for VT.ai LLM service, enabling multi-provider chat and subscription features.",
    version="0.1.0",
)

# Configure CORS
# In production, you should restrict allow_origins to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Example: ["https://your-frontend-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
supabase_url: Optional[str] = os.environ.get("SUPABASE_URL")
supabase_key: Optional[str] = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")
    # Depending on desired behavior, you might raise an error or disable Supabase-dependent features.
    # For now, we'll allow the app to start but log the error.
    supabase: Optional[Client] = None
else:
    try:
        supabase: Optional[Client] = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None

# Initialize Stripe
stripe_secret_key: Optional[str] = os.environ.get("STRIPE_SECRET_KEY")
if not stripe_secret_key:
    logger.error("STRIPE_SECRET_KEY must be set in environment variables.")
    # Similar to Supabase, handle missing Stripe key (e.g., disable payment features)
    # For now, we log the error.
    stripe.api_key = None  # Explicitly set to None if not found
else:
    stripe.api_key = stripe_secret_key
    logger.info("Stripe API key configured.")


# Helper function for Supabase client compatibility
def supabase_query(table_name: str):
    """
    Returns a query builder for the specified table, compatible with both
    sync and async versions of the Supabase client.

    Args:
        table_name: The name of the table to query

    Returns:
        A query builder object for the specified table
    """
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )
    return supabase.table(table_name)


# Pydantic Models
class Message(BaseModel):
    """Represents a single message in a chat conversation."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    messages: List[Message]
    model: str
    max_tokens: Optional[int] = 1000


class SubscriptionRequest(BaseModel):
    """Request model for creating a subscription."""

    price_id: str


class StripeWebhookPayload(BaseModel):
    """Request model for Stripe webhook events."""

    id: str
    object: str
    type: str
    data: dict


# Authentication Middleware
async def get_current_user(authorization: Optional[str] = Header(None)) -> GoTrueUser:
    """
    Authenticates a user based on the Authorization header (Bearer token).

    Args:
            authorization: The Authorization header string, expected to contain a Bearer token.

    Returns:
            The authenticated Supabase user object.

    Raises:
            HTTPException: If authentication fails (e.g., no token, invalid token).
    """
    if supabase is None:
        logger.error("Supabase client not initialized. Authentication unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available.",
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme or token missing. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_response = await supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            # This case might not be hit if get_user throws an error for invalid tokens
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token or user not found.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:  # Catch specific GoTrue/Supabase errors if possible
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token or authentication failed: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# API Key authentication (alternative to Supabase auth)
async def get_api_key_user(
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> dict:
    """
    Authenticates a user based on an API key header.

    This is an alternative authentication method for automated/integration scenarios
    where using a browser-based auth flow isn't practical.

    Args:
            api_key: The API key provided in the X-API-Key header.

    Returns:
            A dict with user information.

    Raises:
            HTTPException: If authentication fails.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing. Please provide the X-API-Key header.",
        )

    if not supabase:
        logger.error(
            "Supabase client not initialized. API key authentication unavailable."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available.",
        )

    try:
        # Query the api_keys table to find the user associated with this key
        response = (
            supabase_query("api_keys")
            .select("user_id, tier, active")
            .eq("key_hash", api_key)
            .eq("active", True)
            .maybe_single()
            .execute()
        )

        if not response.data:
            logger.warning(f"Invalid or inactive API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key.",
            )

        # Return a user-like dictionary with the essential fields
        return {
            "id": response.data["user_id"],
            "api_key_auth": True,
            "tier": response.data.get("tier", "api"),
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"API key authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error.",
        )


# API Routes
@app.post("/api/chat", summary="Process a chat request")
async def chat(
    request: ChatRequest,
    current_user: Optional[GoTrueUser] = Depends(get_current_user),
    api_key_user: Optional[dict] = Depends(get_api_key_user),
) -> dict:
    """
    Handles chat requests, interacting with an LLM provider.

    This endpoint requires either Supabase authentication (Bearer token) or API key authentication.
    It will check the user's subscription tier and token limits before processing the request.

    Args:
            request: The chat request containing messages, model, and other parameters.
            current_user: The authenticated Supabase user object (from Bearer token).
            api_key_user: User information from API key authentication.

    Returns:
            A dictionary containing the LLM's response or an error message.
    """
    # Determine which authentication method was used
    user = None
    user_id = None
    is_api_key_auth = False

    if current_user:
        user = current_user
        user_id = current_user.id
        logger.info(
            f"Chat request received from authenticated user: {user_id} for model: {request.model}"
        )
    elif api_key_user:
        user = api_key_user
        user_id = api_key_user.get("id")
        is_api_key_auth = True
        logger.info(
            f"Chat request received via API key for user: {user_id} for model: {request.model}"
        )
    else:
        # This shouldn't happen due to dependency validation, but just in case
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required."
        )

    # Get user's data including subscription tier
    user_data = get_user_data(user_id)

    # For API key auth, override with the tier stored with the API key
    user_tier = (
        user.get("tier", "free") if is_api_key_auth else user_data.get("tier", "free")
    )

    # Check if user has reached their token limit
    if await has_reached_limit(user_id, user_tier):
        logger.warning(
            f"User {user_id} has reached their token limit for tier: {user_tier}"
        )
        raise HTTPException(
            status_code=429,
            detail="You've reached your token limit for this billing period. Please upgrade your subscription tier.",
        )

    try:
        # Process the request with LiteLLM
        response = await completion(  # type: ignore
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens,
            # Add other LiteLLM parameters as needed (e.g., api_key if not globally set)
        )

        # Track token usage if usage information is available
        if hasattr(response, "usage") and response.usage:
            total_tokens = response.usage.total_tokens
            logger.info(f"Request used {total_tokens} tokens for user {user_id}")
            await track_usage(user_id, total_tokens)
        else:
            logger.warning(
                f"No usage information available for request from user {user_id}"
            )

        # Return the response
        if response.choices and response.choices[0].message:
            return {
                "response": response.choices[0].message.content,
                "usage": response.usage,
                "tier": user_tier,
            }  # type: ignore
        else:
            return {
                "response": "No content in response",
                "usage": response.usage,
                "tier": user_tier,
            }  # type: ignore

    except Exception as e:
        logger.error(
            f"Error during LLM completion for user {user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {e}"
        )


@app.post("/api/create-subscription", summary="Create a new subscription")
async def create_subscription(
    request: SubscriptionRequest, current_user: GoTrueUser = Depends(get_current_user)
) -> dict:
    """
    Creates a new Stripe subscription for the authenticated user.

    This endpoint requires authentication. It will interact with Stripe to
    create a subscription session.

    Args:
            request: The subscription request containing the price ID.
            current_user: The authenticated user object.

    Returns:
            A dictionary containing the Stripe Checkout session ID or an error.
    """
    logger.info(
        f"Create subscription request for user: {current_user.id}, price_id: {request.price_id}"
    )

    if not stripe.api_key:
        logger.error("Stripe API key not configured. Cannot create subscription.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment service is not available.",
        )

    # Get or create a Stripe customer for this user
    stripe_customer_id = await get_or_create_customer(current_user)

    # Get configuration from environment or configuration file
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    success_url = (
        f"{frontend_url}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}"
    )
    cancel_url = f"{frontend_url}/subscription/cancel"

    try:
        # Create a Stripe Checkout session for subscription
        checkout_session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": request.price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=current_user.id,  # Store user ID for webhook processing
            metadata={
                "supabase_user_id": current_user.id,
                "tier": "basic",  # Set subscription tier based on price ID
            },
        )

        logger.info(
            f"Created checkout session {checkout_session.id} for user {current_user.id}"
        )
        return {
            "checkout_session_id": checkout_session.id,
            "checkout_url": checkout_session.url,
        }
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(
            f"Error creating subscription for user {current_user.id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Could not create subscription: {e}"
        )


@app.post("/api/webhook/stripe", summary="Handle Stripe webhook events")
async def stripe_webhook(
    payload: StripeWebhookPayload,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
) -> dict:
    """
    Handles Stripe webhook events for subscription lifecycle management.

    This endpoint receives events from Stripe when subscription-related actions occur,
    such as payment succeeded, subscription created/updated/cancelled, etc.

    Args:
            payload: The webhook payload from Stripe.
            stripe_signature: The Stripe-Signature header for verifying the webhook.

    Returns:
            A dictionary with a success message.
    """
    logger.info(f"Received Stripe webhook event: {payload.type}")

    # Verify webhook signature if STRIPE_WEBHOOK_SECRET is set
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    event = None

    if webhook_secret and stripe_signature:
        try:
            # Construct the event from the payload and signature
            event = stripe.Webhook.construct_event(
                payload.model_dump_json(), stripe_signature, webhook_secret
            )
        except stripe.error.SignatureVerificationError as e:
            logger.error(
                f"Stripe webhook signature verification failed: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=400, detail="Invalid Stripe webhook signature"
            )
    else:
        # For development, we'll trust the payload without verification
        event = payload.model_dump()
        logger.warning(
            "Stripe webhook signature not verified - webhook secret or signature not provided"
        )

    # Process different event types
    event_type = event.get("type", "")

    try:
        if event_type.startswith("customer.subscription"):
            await handle_subscription_event(event)
        elif event_type.startswith("invoice"):
            await handle_invoice_event(event)
        elif event_type == "checkout.session.completed":
            await handle_checkout_completed(event)
        else:
            logger.info(f"Unhandled Stripe webhook event type: {event_type}")

    except Exception as e:
        logger.error(
            f"Error processing Stripe webhook event {event_type}: {e}", exc_info=True
        )
        # We still return 200 to acknowledge receipt to Stripe, but log the error
        # This prevents Stripe from retrying the webhook when the error is on our side

    return {"success": True, "event_type": event_type}


async def handle_subscription_event(event: dict) -> None:
    """
    Handles subscription-related webhook events from Stripe.

    Args:
            event: The Stripe event object.
    """
    event_type = event.get("type", "")
    subscription = event.get("data", {}).get("object", {})
    subscription_id = subscription.get("id")
    customer_id = subscription.get("customer")
    status = subscription.get("status")

    logger.info(
        f"Processing subscription event {event_type} for subscription {subscription_id}"
    )

    if not supabase:
        logger.error(
            "Supabase client not initialized. Cannot process subscription event."
        )
        return

    try:
        # First, find the user associated with this Stripe customer
        response = (
            supabase_query("stripe_customers")
            .select("user_id")
            .eq("stripe_customer_id", customer_id)
            .maybe_single()
            .execute()
        )

        if not response.data:
            logger.error(f"No user found for Stripe customer {customer_id}")
            return

        user_id = response.data.get("user_id")

        # Map Stripe subscription status to our tier system
        tier_mapping = {
            "active": "premium",  # Paid subscription active
            "past_due": "premium",  # Payment failed but still active
            "unpaid": "free",  # Payment failed multiple times
            "canceled": "free",  # Subscription canceled
            "trialing": "premium",  # In trial period
        }

        new_tier = tier_mapping.get(status, "free")

        # Update the user's subscription tier
        supabase_query("user_profiles").update(
            {
                "tier": new_tier,
                "subscription_id": subscription_id,
                "subscription_status": status,
                "updated_at": "now()",
            }
        ).eq("user_id", user_id).execute()

        logger.info(
            f"Updated user {user_id} subscription to {new_tier} tier (status: {status})"
        )

    except Exception as e:
        logger.error(f"Error handling subscription event: {e}", exc_info=True)


async def handle_invoice_event(event: dict) -> None:
    """
    Handles invoice-related webhook events from Stripe.

    Args:
            event: The Stripe event object.
    """
    event_type = event.get("type", "")
    invoice = event.get("data", {}).get("object", {})
    customer_id = invoice.get("customer")

    logger.info(f"Processing invoice event {event_type} for customer {customer_id}")

    # Handle specific invoice events
    if event_type == "invoice.payment_succeeded":
        # Payment was successful - could update payment history or reset counters
        pass
    elif event_type == "invoice.payment_failed":
        # Payment failed - could notify the user
        pass


async def handle_checkout_completed(event: dict) -> None:
    """
    Handles checkout.session.completed webhook events from Stripe.

    Args:
            event: The Stripe event object.
    """
    session = event.get("data", {}).get("object", {})
    user_id = session.get("client_reference_id")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    logger.info(
        f"Processing checkout completed for user {user_id}, subscription {subscription_id}"
    )

    if not user_id or not supabase:
        logger.error("Missing user_id or Supabase client in checkout completed event")
        return

    try:
        # Update user_profiles to reflect the new subscription
        supabase_query("user_profiles").update(
            {
                "tier": "premium",  # Default to premium for new subscriptions
                "subscription_id": subscription_id,
                "subscription_status": "active",
                "updated_at": "now()",
            }
        ).eq("user_id", user_id).execute()

        # Ensure we have the customer mapping
        customer_response = (
            supabase_query("stripe_customers")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        if not customer_response.data:
            # Create the mapping if it doesn't exist
            supabase_query("stripe_customers").insert(
                {
                    "user_id": user_id,
                    "stripe_customer_id": customer_id,
                    "created_at": "now()",
                }
            ).execute()

        logger.info(f"User {user_id} subscription activated: {subscription_id}")

    except Exception as e:
        logger.error(f"Error handling checkout completed: {e}", exc_info=True)


# Placeholder Helper Functions
# These would typically interact with your database (Supabase)


def get_user_data(user_id: str) -> dict:
    """
    Retrieves user-specific data, such as subscription tier.

    Args:
            user_id: The unique identifier of the user.

    Returns:
            A dictionary containing user data.
    """
    logger.info(f"Fetching data for user_id: {user_id}")

    # Default user data if Supabase is not available or user not found
    default_data = {"tier": "free", "user_id": user_id, "tokens_used": 0}

    if not supabase:
        logger.warning(
            f"Supabase client not available. Using default user data for {user_id}"
        )
        return default_data

    try:
        # Fetch user profile from the user_profiles table
        response = (
            supabase_query("user_profiles")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        if response.data:
            logger.debug(f"User data found for {user_id}: {response.data}")
            return response.data
        else:
            # User not found, create a new profile with default values
            logger.info(
                f"User profile not found for {user_id}. Creating default profile."
            )
            insert_response = (
                supabase_query("user_profiles")
                .insert(
                    {
                        "user_id": user_id,
                        "tier": "free",
                        "tokens_used": 0,
                        "created_at": "now()",
                    }
                )
                .execute()
            )

            if insert_response.data:
                return insert_response.data[0]
            return default_data
    except Exception as e:
        logger.error(f"Error fetching user data for {user_id}: {e}", exc_info=True)
        return default_data


async def has_reached_limit(user_id: str, tier: str) -> bool:
    """
    Checks if the user has reached their token/usage limit for their tier.

    Args:
            user_id: The user's unique identifier.
            tier: The user's current subscription tier.

    Returns:
            True if the limit is reached, False otherwise.
    """
    logger.info(f"Checking limits for user_id: {user_id}, tier: {tier}")

    # Define tier-based token limits
    # These could be stored in a configuration file or database table
    tier_limits = {
        "free": 10000,  # 10k tokens for free tier
        "basic": 100000,  # 100k tokens for basic tier
        "premium": 500000,  # 500k tokens for premium tier
        "unlimited": float("inf"),  # No limit for unlimited tier
    }

    # Get default limit for unknown tiers
    default_limit = tier_limits.get("free", 10000)

    # Get the limit for the user's tier
    user_limit = tier_limits.get(tier.lower(), default_limit)

    try:
        # Get user's current usage
        user_data = get_user_data(user_id)
        current_usage = user_data.get("tokens_used", 0)

        # Check if user has reached their limit
        if current_usage >= user_limit:
            logger.warning(
                f"User {user_id} has reached their {tier} tier limit: {current_usage}/{user_limit} tokens"
            )
            return True

        logger.debug(f"User {user_id} usage: {current_usage}/{user_limit} tokens")
        return False
    except Exception as e:
        logger.error(f"Error checking usage limits for {user_id}: {e}", exc_info=True)
        # If we can't check the limit, default to allowing the request
        # This is a business decision - you might want to block instead
        return False


async def track_usage(user_id: str, tokens: int) -> None:
    """
    Tracks token usage for the user.

    Args:
            user_id: The user's unique identifier.
            tokens: The number of tokens used in the current request.
    """
    logger.info(f"Tracking {tokens} tokens for user_id: {user_id}")

    if not supabase:
        logger.warning(
            f"Supabase client not available. Cannot track usage for {user_id}"
        )
        return

    try:
        # First, get the current user data
        user_data = get_user_data(user_id)
        current_tokens = user_data.get("tokens_used", 0)
        new_total = current_tokens + tokens

        # Update the tokens_used field in user_profiles
        update_response = (
            supabase_query("user_profiles")
            .update({"tokens_used": new_total, "last_activity": "now()"})
            .eq("user_id", user_id)
            .execute()
        )

        # Also log this usage record in usage_history for analytics
        supabase_query("usage_history").insert(
            {
                "user_id": user_id,
                "tokens_used": tokens,
                "timestamp": "now()",
                "model": "model_placeholder",  # In a real implementation, pass the model name
            }
        ).execute()

        logger.debug(
            f"Usage tracked for {user_id}: +{tokens} tokens, new total: {new_total}"
        )
    except Exception as e:
        logger.error(f"Error tracking usage for {user_id}: {e}", exc_info=True)
        # Continue execution even if tracking fails - don't block the user experience


async def get_or_create_customer(user: GoTrueUser) -> str:
    """
    Gets an existing Stripe customer ID or creates a new one for the user.

    Args:
            user: The authenticated Supabase user object.

    Returns:
            The Stripe customer ID.
    """
    logger.info(f"Getting/creating Stripe customer for user_id: {user.id}")

    if not supabase:
        logger.error("Supabase client not available. Cannot manage Stripe customers.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    if not stripe.api_key:
        logger.error("Stripe API key not configured. Cannot manage customers.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment service unavailable.",
        )

    try:
        # Check if we already have a Stripe customer ID for this user
        response = (
            supabase_query("stripe_customers")
            .select("stripe_customer_id")
            .eq("user_id", user.id)
            .maybe_single()
            .execute()
        )

        # If found, return the existing Stripe customer ID
        if response.data and "stripe_customer_id" in response.data:
            stripe_customer_id = response.data["stripe_customer_id"]
            logger.debug(
                f"Found existing Stripe customer for {user.id}: {stripe_customer_id}"
            )
            return stripe_customer_id

        # Otherwise, create a new Stripe customer
        logger.info(f"Creating new Stripe customer for user {user.id}")

        # Extract user email from the GoTrueUser object
        user_email = user.email
        if not user_email:
            logger.warning(f"User {user.id} has no email. Using placeholder.")
            user_email = f"user_{user.id}@example.com"  # Placeholder

        # Create the customer in Stripe
        customer = stripe.Customer.create(
            email=user_email,
            name=user.user_metadata.get("full_name", ""),  # If available
            metadata={"supabase_user_id": user.id},
        )

        # Save the mapping to Supabase
        supabase_query("stripe_customers").insert(
            {
                "user_id": user.id,
                "stripe_customer_id": customer.id,
                "created_at": "now()",
            }
        ).execute()

        logger.info(f"Created new Stripe customer for {user.id}: {customer.id}")
        return customer.id

    except stripe.error.StripeError as e:
        logger.error(
            f"Stripe error creating customer for {user.id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(
            f"Error creating Stripe customer for {user.id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to manage customer data.")


# API Key Management Routes
class APIKeyRequest(BaseModel):
    """Request model for creating an API key."""

    description: Optional[str] = "API Key"
    tier: Optional[str] = "free"
    expires_days: Optional[int] = (
        None  # Number of days until expiration, None means never expires
    )


class APIKeyResponse(BaseModel):
    """Response model for API key operations."""

    key: Optional[str] = None  # Only returned when a new key is created
    key_preview: str  # First 8 chars of key hash for display
    id: int
    description: str
    tier: str
    active: bool
    created_at: str
    expires_at: Optional[str] = None


@app.post("/api/keys", summary="Create a new API key", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest, current_user: GoTrueUser = Depends(get_current_user)
) -> APIKeyResponse:
    """
    Creates a new API key for the current user.

    Args:
            request: The API key request with optional description and expiration.
            current_user: The authenticated user.

    Returns:
            The created API key details.
    """
    logger.info(f"Creating new API key for user: {current_user.id}")

    if not supabase:
        logger.error("Supabase client not initialized. Cannot create API key.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    try:
        # Generate a secure random API key
        import datetime
        import hashlib
        import secrets

        api_key = f"vtai_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_preview = key_hash[:8]  # First 8 chars for display

        # Calculate expiration date if specified
        expires_at = None
        if request.expires_days:
            expires_at = datetime.datetime.now() + datetime.timedelta(
                days=request.expires_days
            )

        # Insert the API key into the database
        insert_data = {
            "key_hash": key_hash,
            "user_id": current_user.id,
            "description": request.description,
            "tier": request.tier,
            "active": True,
            "created_at": "now()",
        }

        if expires_at:
            insert_data["expires_at"] = expires_at.isoformat()

        response = supabase_query("api_keys").insert(insert_data).execute()

        if not response.data:
            logger.error(
                f"Error creating API key for user {current_user.id}: No data returned"
            )
            raise HTTPException(status_code=500, detail="Failed to create API key")

        # Return the API key details
        key_data = response.data[0]
        return APIKeyResponse(
            key=api_key,  # Only return the actual key on creation
            key_preview=key_preview,
            id=key_data.get("id"),
            description=key_data.get("description", "API Key"),
            tier=key_data.get("tier", "free"),
            active=key_data.get("active", True),
            created_at=key_data.get("created_at"),
            expires_at=key_data.get("expires_at"),
        )

    except Exception as e:
        logger.error(
            f"Error creating API key for user {current_user.id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {e}")


@app.get(
    "/api/keys", summary="List user's API keys", response_model=List[APIKeyResponse]
)
async def list_api_keys(
    current_user: GoTrueUser = Depends(get_current_user),
) -> List[APIKeyResponse]:
    """
    Lists all API keys for the current user.

    Args:
            current_user: The authenticated user.

    Returns:
            A list of the user's API keys.
    """
    logger.info(f"Listing API keys for user: {current_user.id}")

    if not supabase:
        logger.error("Supabase client not initialized. Cannot list API keys.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    try:
        # Get all API keys for the user
        response = (
            supabase_query("api_keys")
            .select("*")
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .execute()
        )

        if not response.data:
            return []

        # Format the response
        keys = []
        for key_data in response.data:
            key_preview = key_data.get("key_hash", "")[:8]
            keys.append(
                APIKeyResponse(
                    key=None,  # Never return the actual key after creation
                    key_preview=key_preview,
                    id=key_data.get("id"),
                    description=key_data.get("description", "API Key"),
                    tier=key_data.get("tier", "free"),
                    active=key_data.get("active", True),
                    created_at=key_data.get("created_at"),
                    expires_at=key_data.get("expires_at"),
                )
            )

        return keys

    except Exception as e:
        logger.error(
            f"Error listing API keys for user {current_user.id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to list API keys: {e}")


@app.delete("/api/keys/{key_id}", summary="Revoke an API key")
async def revoke_api_key(
    key_id: int, current_user: GoTrueUser = Depends(get_current_user)
) -> dict:
    """
    Revokes (deactivates) an API key.

    Args:
            key_id: The ID of the API key to revoke.
            current_user: The authenticated user.

    Returns:
            A success message.
    """
    logger.info(f"Revoking API key {key_id} for user: {current_user.id}")

    if not supabase:
        logger.error("Supabase client not initialized. Cannot revoke API key.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    try:
        # Verify the key belongs to the user
        key_response = (
            supabase_query("api_keys")
            .select("*")
            .eq("id", key_id)
            .eq("user_id", current_user.id)
            .maybe_single()
            .execute()
        )

        if not key_response.data:
            raise HTTPException(
                status_code=404, detail="API key not found or does not belong to you"
            )

        # Deactivate the key
        supabase_query("api_keys").update({"active": False}).eq("id", key_id).eq(
            "user_id", current_user.id
        ).execute()

        return {"success": True, "message": "API key revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error revoking API key {key_id} for user {current_user.id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to revoke API key: {e}")


# Subscription Management Routes
class SubscriptionPlan(BaseModel):
    """Model for subscription plan details."""

    id: int
    name: str
    stripe_price_id: str
    token_limit: int
    description: Optional[str] = None
    price_usd: float
    active: bool


@app.get("/api/subscription/plans", summary="Get available subscription plans")
async def get_subscription_plans() -> List[SubscriptionPlan]:
    """
    Retrieves all active subscription plans.

    Returns:
            A list of available subscription plans.
    """
    logger.info("Fetching available subscription plans")

    if not supabase:
        logger.error(
            "Supabase client not initialized. Cannot fetch subscription plans."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable.",
        )

    try:
        # Get all active subscription plans
        response = (
            supabase.table("subscription_plans")
            .select("*")
            .eq("active", True)
            .order("price_usd")
            .execute()
        )

        if not response.data:
            logger.warning("No active subscription plans found")
            return []

        # Format and return the plans
        plans = []
        for plan_data in response.data:
            plans.append(
                SubscriptionPlan(
                    id=plan_data.get("id"),
                    name=plan_data.get("name"),
                    stripe_price_id=plan_data.get("stripe_price_id"),
                    token_limit=plan_data.get("token_limit"),
                    description=plan_data.get("description"),
                    price_usd=float(plan_data.get("price_usd", 0)),
                    active=plan_data.get("active", True),
                )
            )

        return plans

    except Exception as e:
        logger.error(f"Error fetching subscription plans: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch subscription plans: {e}"
        )


@app.get("/api/subscription/status", summary="Get user's subscription status")
async def get_subscription_status(
    current_user: GoTrueUser = Depends(get_current_user),
) -> dict:
    """
    Retrieves the current user's subscription status.

    Args:
            current_user: The authenticated user.

    Returns:
            A dictionary with the user's subscription details.
    """
    logger.info(f"Fetching subscription status for user: {current_user.id}")

    try:
        # Get the user's profile
        user_data = get_user_data(current_user.id)

        return {
            "tier": user_data.get("tier", "free"),
            "subscription_id": user_data.get("subscription_id"),
            "subscription_status": user_data.get("subscription_status"),
            "tokens_used": user_data.get("tokens_used", 0),
            "last_activity": user_data.get("last_activity"),
        }

    except Exception as e:
        logger.error(
            f"Error fetching subscription status for user {current_user.id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch subscription status: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server for VT.ai API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
