from pydantic import Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from deepeval.test_case import LLMTestCase


@dataclass
class ConversationalTestCase:
    messages: List[LLMTestCase]
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.messages) == 0:
            raise TypeError("'messages' must not be empty")

        if not isinstance(self.messages, list) or not all(
            isinstance(item, LLMTestCase) for item in self.messages
        ):
            raise TypeError("'messages' must be a list of LLMTestCase")
