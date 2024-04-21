import chainlit as cl


def set_single_user_session_value(key):
    cl.user_session.set(key, key)


def set_user_session_value(key, value):
    cl.user_session.set(key, value)


def get_user_session_value(key):
    return cl.user_session.get(key)
