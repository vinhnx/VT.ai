from unittest.mock import patch

import pytest

from vtai.utils.credits import DEFAULT_DAILY_CREDITS, get_user_credits_info


@pytest.mark.integration
def test_user_profile_credits_integration():
    user_id = "test_user_123"
    # Simulate 4 successful requests in analytics
    with patch("vtai.utils.credits.get_user_analytics") as mock_analytics:
        mock_analytics.return_value = {"successful_requests": 4}
        credits = get_user_credits_info(user_id)
        assert credits["credits_left"] == DEFAULT_DAILY_CREDITS - 4
        assert credits["max_credits"] == DEFAULT_DAILY_CREDITS
        assert "reset_time" in credits

    # Simulate user with no usage
    with patch("vtai.utils.credits.get_user_analytics") as mock_analytics:
        mock_analytics.return_value = {"successful_requests": 0}
        credits = get_user_credits_info(user_id)
        assert credits["credits_left"] == DEFAULT_DAILY_CREDITS

    # Simulate user with maxed out usage
    with patch("vtai.utils.credits.get_user_analytics") as mock_analytics:
        mock_analytics.return_value = {"successful_requests": DEFAULT_DAILY_CREDITS}
        credits = get_user_credits_info(user_id)
        assert credits["credits_left"] == 0
        mock_analytics.return_value = {"successful_requests": DEFAULT_DAILY_CREDITS}
        credits = get_user_credits_info(user_id)
        assert credits["credits_left"] == 0
