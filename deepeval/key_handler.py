"""File for handling API key
"""
import json

from .constants import KEY_FILE


class KeyFileHandler:
    def __init__(self):
        self.data = {}

    def write_data(self, key, value):
        """Writes data to the hidden file"""
        self.data[key] = value
        with open(KEY_FILE, "w") as f:
            json.dump(self.data, f)

    def fetch_data(self, key):
        """Fetches the data from the hidden file"""
        try:
            with open(KEY_FILE, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            # Handle the case when the file doesn't exist
            self.data = {}
        return self.data.get(key)

    def write_api_key(self, api_key):
        """Writes the API key to the hidden file"""
        self.write_data("api_key", api_key)

    def fetch_api_key(self) -> str:
        """Fetches the API key from the hidden file"""
        return self.fetch_data("api_key")


KEY_FILE_HANDLER = KeyFileHandler()
