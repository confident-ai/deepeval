from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


@dataclass
class Message:
    user_input: str
    llm_response: str
    retrieval_context: Optional[List[str]] = None

    def __post_init__(self):
        # Ensure `retrieval_context` is None or a list of strings
        if self.retrieval_context is not None:
            if not isinstance(self.retrieval_context, list) or not all(
                isinstance(item, str) for item in self.retrieval_context
            ):
                raise TypeError(
                    "retrieval_context must be None or a list of strings"
                )


class ConversationalTestCaseParams(Enum):
    USER_INPUT = "input"
    LLM_RESPONSE = "response"


@dataclass
class ConversationalTestCase:
    messages: List[Message]
