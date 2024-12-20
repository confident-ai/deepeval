import os
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum


@dataclass
class MLLMImage:
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
        if parsed_url.scheme == "file" or parsed_url.scheme == "":
            # Check if the path exists on the filesystem
            return os.path.exists(parsed_url.path)

        return False


class MLLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"


@dataclass
class MLLMTestCase:
    input: List[Union[str, MLLMImage]]
    actual_output: List[Union[str, MLLMImage]]
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    name: Optional[str] = field(default=None)
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)
