from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Optional
import os

class Languages(Enum):
    ENGLISH = "English"
    SPANISH = "Spanish"

@dataclass
class Image:
    url: str
    local: Optional[bool] = None

    def __post_init__(self):
        if self.local == None:
            self.local = self.is_local_path(self.url)

    @staticmethod
    def is_local_path(url):
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Check if it's a file scheme or an empty scheme with a local path
        if parsed_url.scheme == 'file' or parsed_url.scheme == '':
            # Check if the path exists on the filesystem
            return os.path.exists(parsed_url.path)
        
        return False
