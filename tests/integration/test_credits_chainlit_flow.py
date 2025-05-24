from unittest.mock import MagicMock, patch

import chainlit as cl
import pytest

from vtai.app import send_user_profile_async
from vtai.utils.credits import DEFAULT_DAILY_CREDITS, get_user_credits_info


@pytest.mark.integration
def test_chainlit_profile_credits_flow(monkeypatch):
    user_id = "test_chainlit_user"
    # Patch Chainlit user session
    monkeypatch.setattr(cl, "user_session", {"user_id": user_id})

    # Simulate 2 successful requests in analytics
    with patch("vtai.utils.credits.get_user_analytics") as mock_analytics:
        mock_analytics.return_value = {"successful_requests": 2}

        # Patch cl.Message.send to capture output
        sent_msgs = []

        class DummyMsg:
            def __init__(self, content=None, elements=None):
                self.content = content
                self.elements = elements

            async def send(self):
                sent_msgs.append(self)

        monkeypatch.setattr(cl, "Message", DummyMsg)

        # Call the async profile sender
        import asyncio

        asyncio.run(send_user_profile_async({"user_id": user_id}))

        # Check that credits info is present in the sent message
        assert sent_msgs
        profile_props = sent_msgs[0].elements[0].props
        assert profile_props["credits_left"] == DEFAULT_DAILY_CREDITS - 2
        assert profile_props["max_credits"] == DEFAULT_DAILY_CREDITS
        profile_props = sent_msgs[0].elements[0].props
        assert profile_props["credits_left"] == DEFAULT_DAILY_CREDITS - 2
        assert profile_props["max_credits"] == DEFAULT_DAILY_CREDITS
