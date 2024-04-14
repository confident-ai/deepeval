from dataclasses import dataclass
from typing import List, Optional
from deepeval.test_case import LLMTestCase


@dataclass
class ConversationalTestCase:
    messages: List[LLMTestCase]
    dataset_alias: Optional[str] = None
