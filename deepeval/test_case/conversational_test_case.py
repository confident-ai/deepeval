from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from .llm_test_case import LLMTestCase


@dataclass
class ConversationalTestCase:
    messages: List[LLMTestCase]
