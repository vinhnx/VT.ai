"""
Stripe Webhook Handler for VT.ai.

This module provides an API endpoint to receive and process webhooks from Stripe.
It handles events such as successful payments, subscription updates, cancellations,
and updates the user's subscription status in the Supabase database accordingly.
"""
import os
import stripe
from fastapi import APIRouter, Request, HTTPException, Header, Depends, status
from supabase import Client, create_client
import dotenv

# Attempt to import the configured logger
try:
    from vtai.utils.config import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Initialize Supabase
supabase_url: str | None = os.environ.get("SUPABASE_URL")
supabase_key: str | None = os.environ.get("SUPABASE_KEY")
stripe_webhook_secret: str | None = os.environ.get("STRIPE_WEBHOOK_SECRET")

if not supabase_url or not supabase_key:
    logger.error("Supabase URL or Key not found for webhook handler.")
    supabase: Client | None = None
else:
    try:
        supabase: Client | None = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully for webhook handler.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client for webhook handler: {e}")
        supabase = None

if not stripe_webhook_secret:
    logger.warning("STRIPE_WEBHOOK_SECRET is not set. Webhook signature verification will be skipped.")

# Initialize router
router = APIRouter(tags=["webhooks"])

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
        logger.error("Supabase client is not initialized")
        return None
    return supabase.table(table_name)

# Helper function to map Stripe subscription status to your application's tier or status
def get_tier_from_subscription(subscription: stripe.Subscription) -> str:
    """
    Determines the application-specific tier based on a Stripe subscription object.

    Args:
        subscription: The Stripe Subscription object.

    Returns:
        A string representing the user's tier (e.g., 'pro', 'enterprise', 'free').
    """
    if not subscription or not subscription.items or not subscription.items.data:
        return "free"  # Default to free if subscription data is incomplete

    # Assuming the price ID or a metadata field on the price/product determines the tier.
    # This is a placeholder. You need to map your Stripe Price IDs to your tiers.
    price_id = subscription.items.data[0].price.id
    # Example mapping (replace with your actual Price IDs and tiers)
    if price_id == "price_pro_tier_id_from_stripe": # Replace with actual Stripe Price ID
        return "pro"
    elif price_id == "price_enterprise_tier_id_from_stripe": # Replace with actual Stripe Price ID
        return "enterprise"
    else:
        # If it's an active subscription but unknown price_id, or if you want to handle other cases
        if subscription.status == "active":
            logger.warning(f"Unknown Price ID {price_id} for active subscription {subscription.id}. Defaulting tier.")
            # Potentially default to a basic paid tier or investigate
            return "pro" # Or some other default paid tier
        return "free" # Default for non-active or unmapped subscriptions

def update_user_subscription_status(
    stripe_customer_id: str,
    tier: str,
    subscription_status: str,
    subscription_id: str | None = None
) -> None:
    """
    Updates the user's subscription status in the Supabase 'profiles' table.

    Args:
        stripe_customer_id: The Stripe customer ID.
        tier: The new subscription tier for the user.
        subscription_status: The status of the subscription (e.g., 'active', 'canceled').
        subscription_id: The Stripe subscription ID.
    """
    if not supabase:
        logger.error("Supabase client not available. Cannot update user subscription.")
        return

    try:
        update_data = {
            "subscription_tier": tier,
            "stripe_subscription_id": subscription_id, # Assuming you add this column to profiles
            "stripe_subscription_status": subscription_status # Assuming you add this column
        }
        # Remove None values to avoid overwriting with NULL if not provided
        update_data = {k: v for k, v in update_data.items() if v is not None}

        response = (
            supabase_query("profiles")
            .update(update_data)
            .eq("stripe_customer_id", stripe_customer_id)
            .execute()
        )
        if response.data:
            logger.info(f"Successfully updated subscription for Stripe customer {stripe_customer_id} to tier {tier}, status {subscription_status}.")
        else:
            logger.error(f"Failed to update subscription for Stripe customer {stripe_customer_id}. Response: {response.error or 'No data returned'}")
    except Exception as e:
        logger.error(f"Error updating Supabase for Stripe customer {stripe_customer_id}: {e}", exc_info=True)

@router.post("/webhooks/stripe", summary="Handle Stripe webhook events")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(None)):
    """
    Endpoint to receive and process webhooks from Stripe.

    It verifies the webhook signature (if STRIPE_WEBHOOK_SECRET is set)
    and processes relevant events like `invoice.paid`, `customer.subscription.updated`,
    `customer.subscription.deleted` to update user subscription status in Supabase.

    Args:
        request: The FastAPI Request object containing the webhook payload.
        stripe_signature: The value of the 'Stripe-Signature' header.

    Returns:
        A success message or an HTTP error response.
    """
    payload = await request.body()
    event = None

    if not stripe.api_key:
        logger.error("Stripe API key not configured. Cannot process webhooks.")
        raise HTTPException(status_code=503, detail="Stripe service not configured.")

    if stripe_webhook_secret and stripe_signature:
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, stripe_webhook_secret
            )
            logger.info(f"Stripe webhook event received and verified: {event.type}")
        except ValueError as e:
            # Invalid payload
            logger.error(f"Invalid Stripe webhook payload: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            # Invalid signature
            logger.error(f"Invalid Stripe webhook signature: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid signature")
    elif stripe_webhook_secret and not stripe_signature:
        logger.warning("Stripe webhook secret is configured, but no signature was provided with the request.")
        raise HTTPException(status_code=400, detail="Stripe-Signature header missing.")
    else:
        # No secret configured, attempt to parse event directly (less secure)
        try:
            event = stripe.Event.construct_from(stripe.json.loads(payload), stripe.api_key)
            logger.warning(f"Stripe webhook event received (NO SIGNATURE VERIFICATION): {event.type}")
        except Exception as e:
            logger.error(f"Error constructing event from payload (no verification): {e}")
            raise HTTPException(status_code=400, detail="Invalid payload or event construction error.")

    if not event:
        logger.error("Stripe event could not be constructed or verified.")
        raise HTTPException(status_code=500, detail="Could not process webhook event.")

    # Handle the event
    if event.type == 'invoice.paid':
        invoice = event.data.object
        if invoice.billing_reason == 'subscription_create' or invoice.billing_reason == 'subscription_cycle':
            stripe_customer_id = invoice.customer
            subscription_id = invoice.subscription
            if stripe_customer_id and subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    tier = get_tier_from_subscription(subscription)
                    update_user_subscription_status(stripe_customer_id, tier, "active", subscription_id)
                    logger.info(f"Processed invoice.paid for customer {stripe_customer_id}, subscription {subscription_id}, tier {tier}.")
                except stripe.error.StripeError as e:
                    logger.error(f"Stripe API error retrieving subscription {subscription_id}: {e}")
                except Exception as e:
                    logger.error(f"Error processing invoice.paid for {stripe_customer_id}: {e}", exc_info=True)
            else:
                logger.warning(f"Invoice.paid event {invoice.id} missing customer or subscription ID.")

    elif event.type == 'customer.subscription.updated':
        subscription = event.data.object
        stripe_customer_id = subscription.customer
        if stripe_customer_id:
            tier = get_tier_from_subscription(subscription)
            update_user_subscription_status(stripe_customer_id, tier, subscription.status, subscription.id)
            logger.info(f"Processed customer.subscription.updated for customer {stripe_customer_id}, status {subscription.status}, tier {tier}.")
        else:
            logger.warning(f"customer.subscription.updated event {subscription.id} missing customer ID.")

    elif event.type == 'customer.subscription.deleted': # Handles cancellations
        subscription = event.data.object
        stripe_customer_id = subscription.customer
        if stripe_customer_id:
            # When a subscription is deleted, user might revert to 'free' tier
            update_user_subscription_status(stripe_customer_id, "free", "canceled", subscription.id)
            logger.info(f"Processed customer.subscription.deleted for customer {stripe_customer_id}. User set to free tier.")
        else:
            logger.warning(f"customer.subscription.deleted event {subscription.id} missing customer ID.")

    # Add other event types as needed (e.g., payment_failed, subscription_trial_ending)
    else:
        logger.info(f"Received unhandled Stripe event type: {event.type}")

    return {"status": "success", "event_received": event.type}

# To include this router in your main FastAPI app:
# from vtai.webhooks import router as webhook_router
# app.include_router(webhook_router)