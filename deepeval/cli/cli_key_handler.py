import os
from ..key_handler import KEY_FILE_HANDLER
from ..constants import API_KEY_ENV, IMPLEMENTATION_ID_NAME


def set_api_key():
    api_key = KEY_FILE_HANDLER.fetch_api_key()
    os.environ[API_KEY_ENV] = api_key


def set_implementation_name():
    imp_name = KEY_FILE_HANDLER.fetch_implementation_name()
    os.environ[IMPLEMENTATION_ID_NAME] = imp_name


def set_env_vars():
    set_api_key()
    set_implementation_name()
