from unittest.mock import patch

import pytest

from vtai.utils.credits import (
    DEFAULT_DAILY_CREDITS,
    check_user_can_chat,
    get_user_credits,
)


def fake_analytics(successful_requests):
    return {"successful_requests": successful_requests}


@patch("vtai.utils.credits.get_user_analytics")
def test_get_user_credits_basic(mock_analytics):
    mock_analytics.return_value = fake_analytics(3)
    result = get_user_credits("user1")
    assert result["credits_left"] == DEFAULT_DAILY_CREDITS - 3
    assert result["max_credits"] == DEFAULT_DAILY_CREDITS
    assert "reset_time" in result


@patch("vtai.utils.credits.get_user_analytics")
def test_get_user_credits_zero(mock_analytics):
    mock_analytics.return_value = fake_analytics(DEFAULT_DAILY_CREDITS)
    result = get_user_credits("user2")
    assert result["credits_left"] == 0


@patch("vtai.utils.credits.get_user_analytics")
def test_check_user_can_chat_true(mock_analytics):
    mock_analytics.return_value = fake_analytics(2)
    assert check_user_can_chat("user3") is True


@patch("vtai.utils.credits.get_user_analytics")
def test_check_user_can_chat_false(mock_analytics):
    mock_analytics.return_value = fake_analytics(DEFAULT_DAILY_CREDITS)
    assert check_user_can_chat("user4") is False
    mock_analytics.return_value = fake_analytics(2)
    assert check_user_can_chat("user3") is True


@patch("vtai.utils.credits.get_user_analytics")
def test_check_user_can_chat_false(mock_analytics):
    mock_analytics.return_value = fake_analytics(DEFAULT_DAILY_CREDITS)
    assert check_user_can_chat("user4") is False
