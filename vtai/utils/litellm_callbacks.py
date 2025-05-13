"""
LiteLLM callbacks for token usage tracking.

This module provides custom callback handlers for LiteLLM to track token usage
across different LLM providers and store the data in Supabase.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chainlit as cl
import litellm
from supabase import Client

logger = logging.getLogger("vt.ai")


class VTAISupabaseHandler:
    """
    Custom Supabase callback handler for VT.ai token usage tracking.
    Implements a callback handler for LiteLLM to log usage data to Supabase.
    """

    def __init__(
        self,
        supabase_client: Client,
        table_name: str = "request_logs",
        log_to_legacy: bool = True,
    ):
        """
        Initialize the VT.ai Supabase callback handler.

        Args:
            supabase_client: The initialized Supabase client
            table_name: The Supabase table name to log to
            log_to_legacy: Whether to also log to the legacy usage_logs table
        """
        self.client = supabase_client
        self.table_name = table_name
        self.log_to_legacy = log_to_legacy

    def __call__(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any = None,
        start_time: float = None,
        end_time: float = None,
        error: Any = None,
    ) -> None:
        """
        General callback method for compatibility with different LiteLLM versions.
        This dispatches to the appropriate specialized method based on the arguments.

        Args:
            kwargs: The arguments passed to the LLM call
            response_obj: The response from the LLM call (for success)
            start_time: The time when the call started
            end_time: The time when the call ended
            error: The error that occurred (for failure)
        """
        # Log this call for debugging
        logger.info("LiteLLM callback triggered via __call__ method")

        # Route to the appropriate handler
        if error is not None:
            self.log_failure_event(kwargs, error, start_time, end_time)
        else:
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Log successful LLM API calls to Supabase.

        Args:
            kwargs: The arguments passed to the LLM call
            response_obj: The response from the LLM call
            start_time: The time when the call started
            end_time: The time when the call ended
        """
        try:
            logger.info(
                "LiteLLM success callback triggered - logging tokens to Supabase"
            )

            # Get user ID from session if available
            user_id = self._get_user_id_from_session(kwargs)
            logger.debug("Extracted user_id: %s", user_id)

            # Extract relevant information from the request and response
            model_name = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            # Get session ID or generate a unique one
            session_id = kwargs.get("litellm_params", {}).get("session_id", "unknown")
            if session_id == "unknown":
                session_id = kwargs.get("metadata", {}).get(
                    "session_id", f"session-{str(uuid.uuid4())}"
                )

            # Calculate response time in seconds
            response_time = end_time - start_time

            # Extract token usage
            input_tokens = getattr(response_obj, "usage", {}).get("prompt_tokens", 0)
            output_tokens = getattr(response_obj, "usage", {}).get(
                "completion_tokens", 0
            )
            total_tokens = getattr(response_obj, "usage", {}).get("total_tokens", 0)

            # If token counts are not in the response, try to estimate them
            if total_tokens == 0 and messages:
                try:
                    # Simple token estimation based on message length
                    input_tokens = sum(
                        len(m.get("content", "").split()) for m in messages
                    )
                    # Approximate output tokens from response length
                    if (
                        hasattr(response_obj, "choices")
                        and len(response_obj.choices) > 0
                    ):
                        response_text = response_obj.choices[0].message.content or ""
                        output_tokens = (
                            len(response_text.split()) * 1.3
                        )  # Rough estimate
                    total_tokens = input_tokens + output_tokens
                except Exception as e:
                    logger.debug(f"Error estimating token count: {e}")

            # Extract cost if available
            cost = getattr(response_obj, "cost", None)

            # Generate a unique ID for this call
            call_id = kwargs.get("litellm_params", {}).get(
                "litellm_call_id", f"call-{str(uuid.uuid4())}"
            )

            # Format the response for storage
            response_to_store = {}
            try:
                if hasattr(response_obj, "model_dump"):
                    response_to_store = response_obj.model_dump()  # For Pydantic models
                elif hasattr(response_obj, "to_dict"):
                    response_to_store = response_obj.to_dict()
                else:
                    # Extract common fields
                    if hasattr(response_obj, "id"):
                        response_to_store["id"] = response_obj.id
                    if hasattr(response_obj, "choices"):
                        response_to_store["choices"] = [
                            {
                                "index": c.index,
                                "message": {
                                    "role": c.message.role,
                                    "content": c.message.content,
                                },
                            }
                            for c in response_obj.choices
                        ]
                    if hasattr(response_obj, "usage"):
                        response_to_store["usage"] = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total_tokens,
                        }
            except Exception as e:
                logger.debug(f"Error formatting response for storage: {e}")
                response_to_store = {"error": "Could not format response for storage"}

            # Prepare additional details
            additional_details = {
                "session_id": session_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "user_id": user_id,
                "metadata": kwargs.get("metadata", {}),
            }

            # Prepare the data for insertion into the request_logs table
            request_log_entry = {
                "model": model_name,
                "messages": json.dumps(messages),
                "response": json.dumps(response_to_store),
                "end_user": user_id,
                "status": "success",
                "response_time": response_time,
                "total_cost": cost if cost is not None else 0.0,
                "additional_details": json.dumps(additional_details),
                "litellm_call_id": call_id,
            }

            # Insert the data into the request_logs table
            try:
                # Create the query without awaiting it
                query = self.client.table(self.table_name).insert(request_log_entry)

                # Execute the query directly (not awaited)
                result = query.execute()

                if hasattr(result, "error") and result.error:
                    logger.error(
                        f"Error inserting into {self.table_name} table: {result.error}"
                    )
                else:
                    logger.debug(
                        f"Successfully logged to {self.table_name} table for user {user_id}"
                    )
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error(
                        "Row-level security policy prevented logging usage to request_logs table. "
                        "This may happen if the user doesn't have the right permissions."
                    )
                    logger.error(
                        "Consider using SUPABASE_SERVICE_KEY instead of SUPABASE_KEY for token tracking."
                    )
                else:
                    logger.error(f"Error inserting into {self.table_name} table: {e}")

            # Log to legacy usage_logs table if enabled
            if self.log_to_legacy and user_id:
                self._log_to_legacy_table(
                    user_id=user_id,
                    session_id=session_id,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                )

        except Exception as e:
            logger.error(f"Error in VTAISupabaseHandler.log_success_event: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def log_failure_event(
        self, kwargs: Dict[str, Any], error: Any, start_time: float, end_time: float
    ) -> None:
        """Log failed LLM API calls to Supabase.

        Args:
            kwargs: The arguments passed to the LLM call
            error: The error that occurred
            start_time: The time when the call started
            end_time: The time when the call ended
        """
        try:
            # Get user ID from session if available
            user_id = self._get_user_id_from_session(kwargs)

            # Extract relevant information from the request
            model_name = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            # Get session ID or generate a unique one
            session_id = kwargs.get("litellm_params", {}).get("session_id", "unknown")
            if session_id == "unknown":
                session_id = kwargs.get("metadata", {}).get(
                    "session_id", f"session-{str(uuid.uuid4())}"
                )

            # Calculate response time in seconds
            response_time = end_time - start_time

            # Generate a unique ID for this call
            call_id = kwargs.get("litellm_params", {}).get(
                "litellm_call_id", f"call-{str(uuid.uuid4())}"
            )

            # Format the error for storage
            error_to_store = {}
            if isinstance(error, Exception):
                error_to_store = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                }
            else:
                error_to_store = {"message": str(error)}

            # Prepare additional details
            additional_details = {
                "session_id": session_id,
                "user_id": user_id,
                "metadata": kwargs.get("metadata", {}),
            }

            # Prepare the data for insertion into the request_logs table
            request_log_entry = {
                "model": model_name,
                "messages": json.dumps(messages),
                "end_user": user_id,
                "status": "failed",
                "error": json.dumps(error_to_store),
                "response_time": response_time,
                "additional_details": json.dumps(additional_details),
                "litellm_call_id": call_id,
            }

            # Insert the data into the request_logs table
            try:
                # Create the query without awaiting it
                query = self.client.table(self.table_name).insert(request_log_entry)

                # Execute the query directly (not awaited)
                result = query.execute()

                if hasattr(result, "error") and result.error:
                    logger.error(
                        f"Error inserting into {self.table_name} table: {result.error}"
                    )
                else:
                    logger.debug(
                        f"Successfully logged failure to {self.table_name} table for user {user_id}"
                    )
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error(
                        "Row-level security policy prevented logging failure event to request_logs table. "
                        "This may happen if the user doesn't have the right permissions."
                    )
                    logger.error(
                        "Consider using SUPABASE_SERVICE_KEY instead of SUPABASE_KEY for token tracking."
                    )
                else:
                    logger.error(
                        f"Error inserting failure event into {self.table_name} table: {e}"
                    )

        except Exception as e:
            logger.error(f"Error in VTAISupabaseHandler.log_failure_event: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def _get_user_id_from_session(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Get the user ID from the session or from the kwargs.

        Args:
            kwargs: The kwargs from the LiteLLM call

        Returns:
            The user ID if available, None otherwise
        """
        # Try to get user ID from kwargs (explicitly passed by caller)
        user_id = kwargs.get("user")

        # If not found in kwargs, try to get from Chainlit user session
        if not user_id:
            try:
                # Check if authenticated via Chainlit's password auth
                chainlit_user = cl.user_session.get("user")
                if chainlit_user:
                    if isinstance(chainlit_user, cl.User):
                        # Password authenticated user
                        user_id = chainlit_user.identifier
                    elif isinstance(chainlit_user, dict) and "id" in chainlit_user:
                        # Supabase authenticated user
                        user_id = chainlit_user.get("id")
                    elif hasattr(chainlit_user, "id"):
                        # Object with ID attribute
                        user_id = chainlit_user.id
            except Exception as e:
                logger.debug(f"Error getting user ID from Chainlit session: {e}")
                # Continue with None user_id

        return user_id

    def _log_to_legacy_table(
        self,
        user_id: str,
        session_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cost: Optional[float] = None,
    ):
        """
        Log token usage to the legacy usage_logs table.

        Args:
            user_id: The user ID
            session_id: The session ID
            model_name: The model name
            input_tokens: The number of input tokens
            output_tokens: The number of output tokens
            total_tokens: The total number of tokens
            cost: The cost of the request if available
        """
        try:
            # Determine auth method
            auth_method = "supabase"

            # Try to get user from Chainlit session
            chainlit_user = cl.user_session.get("user")
            if (
                chainlit_user
                and hasattr(chainlit_user, "metadata")
                and chainlit_user.metadata
            ):
                if chainlit_user.metadata.get("provider") == "credentials":
                    auth_method = "password"
                elif chainlit_user.metadata.get("oauth") == True:
                    auth_method = f"{chainlit_user.metadata.get('provider', 'oauth')}"

            # Format the current time in ISO 8601 format
            current_time = datetime.utcnow().isoformat() + "Z"

            # Prepare the data for the legacy table
            legacy_data = {
                "user_id": user_id,
                "session_id": session_id,
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost if cost is not None else 0.0,
                "created_at": current_time,
                "auth_method": auth_method,
            }

            # Insert the data into the legacy table
            try:
                # Create the query without awaiting it
                query = self.client.table("usage_logs").insert(legacy_data)

                # Execute the query directly (not awaited)
                result = query.execute()

                if hasattr(result, "error") and result.error:
                    logger.error(
                        f"Error inserting into legacy usage_logs table: {result.error}"
                    )
                else:
                    logger.debug(
                        f"Successfully logged to legacy usage_logs table for user {user_id}"
                    )
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error(
                        "Row-level security policy prevented logging to legacy usage_logs table. "
                        "This may happen if the user doesn't have the right permissions."
                    )
                    logger.error(
                        "Consider using SUPABASE_SERVICE_KEY instead of SUPABASE_KEY for token tracking."
                    )
                else:
                    logger.error(f"Error inserting into legacy usage_logs table: {e}")

        except Exception as e:
            logger.error(f"Error logging to legacy usage_logs table: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")


def initialize_litellm_callbacks(
    supabase_client: Optional[Client] = None,
    log_to_legacy: bool = True,
) -> None:
    """
    Initialize LiteLLM callbacks for token usage tracking.

    Args:
        supabase_client: The initialized Supabase client
        log_to_legacy: Whether to also log to the legacy usage_logs table
    """
    if not supabase_client:
        logger.warning(
            "Supabase client not provided. Token usage tracking will be disabled."
        )
        return

    try:
        # Configure the callback handler
        callback_handler = VTAISupabaseHandler(
            supabase_client=supabase_client,
            table_name="request_logs",
            log_to_legacy=log_to_legacy,
        )

        # Register the callback handler with LiteLLM
        litellm.callbacks = [callback_handler]

        # Set success and failure callback handlers
        # In newer versions of LiteLLM, this is how callbacks are configured
        if hasattr(litellm, "success_callback"):
            litellm.success_callback = [callback_handler]
        if hasattr(litellm, "failure_callback"):
            litellm.failure_callback = [callback_handler]

        logger.info(
            "Successfully initialized LiteLLM callbacks for token usage tracking"
        )

    except Exception as e:
        logger.error(f"Error initializing LiteLLM callbacks: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


async def log_usage_to_supabase(
    user_id: Optional[str],
    session_id: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost: Optional[float] = None,
) -> None:
    """
    Manually log token usage to Supabase.
    This function can be used when the automatic callbacks fail or for scenarios not covered by callbacks.

    Args:
        user_id: The user ID
        session_id: The session ID
        model_name: The model name
        input_tokens: The number of input tokens
        output_tokens: The number of output tokens
        total_tokens: The total number of tokens
        cost: The cost of the request if available
    """
    # Import supabase_client locally to avoid circular imports
    try:
        from vtai.app import supabase_client
    except ImportError:
        try:
            from vtai.main import supabase_client
        except ImportError:
            logger.error(
                "Could not import supabase_client. Manual token usage logging will be disabled."
            )
            return

    if not supabase_client:
        logger.warning(
            "Supabase client not available. Manual token usage logging will be disabled."
        )
        return

    logger.debug(
        f"Logging usage to request_logs table - Model: {model_name}, User: {user_id or 'anonymous'}, Session: {session_id}"
    )
    logger.debug(
        f"Token counts - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}"
    )

    # Check if user is authenticated with built-in password system rather than Supabase
    user = cl.user_session.get("user")
    if (
        user
        and hasattr(user, "metadata")
        and user.metadata.get("provider") == "credentials"
    ):
        logger.info("User authenticated via password auth. Skipping usage logging.")
        return

    try:
        # Format the data for the request_logs table structure
        request_log_entry = {
            "model": model_name,
            "messages": json.dumps([]),  # Empty messages array
            "end_user": user_id,
            "status": "success",
            "response_time": 0,
            "total_cost": cost if cost is not None else 0.0,
            "additional_details": json.dumps(
                {
                    "session_id": session_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "manual_log": True,
                }
            ),
            "litellm_call_id": f"manual-{int(time.time())}-{session_id}",  # Generate a unique ID
        }

        # Insert data into request_logs table
        try:
            # Create the query without awaiting it
            query = supabase_client.table("request_logs").insert(request_log_entry)

            # Execute the query directly (not awaited)
            result = query.execute()

            if hasattr(result, "error") and result.error:
                logger.error(f"Error inserting into request_logs table: {result.error}")
            else:
                logger.debug(
                    f"Successfully logged to request_logs table for user {user_id}"
                )
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error(
                    "Row-level security policy prevented logging to request_logs table. "
                    "This may happen if the user doesn't have the right permissions."
                )
                logger.error(
                    "Consider using SUPABASE_SERVICE_KEY instead of SUPABASE_KEY for token tracking."
                )
            else:
                logger.error(f"Error inserting into request_logs table: {e}")

        # Also log to legacy usage_logs table
        legacy_data = {
            "user_id": user_id,
            "session_id": session_id,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost if cost is not None else 0.0,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "auth_method": "supabase",
        }

        # Insert into legacy table
        try:
            # Create the query without awaiting it
            legacy_query = supabase_client.table("usage_logs").insert(legacy_data)

            # Execute the query directly (not awaited)
            legacy_result = legacy_query.execute()

            if hasattr(legacy_result, "error") and legacy_result.error:
                logger.error(
                    f"Error inserting into legacy usage_logs table: {legacy_result.error}"
                )
            else:
                logger.debug(
                    f"Successfully logged to legacy usage_logs table for user {user_id}"
                )
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error(
                    "Row-level security policy prevented logging to legacy usage_logs table. "
                    "This may happen if the user doesn't have the right permissions."
                )
            else:
                logger.error(f"Error inserting into legacy usage_logs table: {e}")

    except Exception as e:
        logger.error(f"Error logging usage to Supabase: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
