"""Utility function for getting API key.
"""
import os

from .constants import API_KEY_ENV
from .key_handler import KEY_FILE_HANDLER


def _get_api_key():
    """Get an API key here."""
    api_key = KEY_FILE_HANDLER.fetch_api_key()
    if api_key is None or api_key == "":
        api_key = os.getenv(API_KEY_ENV, "")
    return api_key
