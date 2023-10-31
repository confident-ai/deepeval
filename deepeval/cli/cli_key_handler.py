import os

from ..constants import API_KEY_ENV
from ..key_handler import KEY_FILE_HANDLER


def set_api_key():
    api_key = KEY_FILE_HANDLER.fetch_api_key()
    os.environ[API_KEY_ENV] = api_key


def set_env_vars():
    set_api_key()
