"""Utility function for getting API key.
"""
import os
from .key_handler import KEY_FILE_HANDLER
from .constants import API_KEY_ENV, IMPLEMENTATION_ID_NAME


def _get_api_key():
    """Get an API key here."""
    api_key = KEY_FILE_HANDLER.fetch_api_key()
    if api_key is None or api_key == "":
        api_key = os.getenv(API_KEY_ENV, "")
    return api_key


def _get_implementation_name():
    imp_name = KEY_FILE_HANDLER.fetch_implementation_name()
    if imp_name is None or imp_name == "":
        imp_name = os.getenv(IMPLEMENTATION_ID_NAME, "")
    return imp_name
