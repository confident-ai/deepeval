from pydantic import Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum

from deepeval.types import Image

class MLLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"

@dataclass
class MLLMTestCase:
    input: List[Union[str, Image]]
    actual_output: List[Union[str, Image]]
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)