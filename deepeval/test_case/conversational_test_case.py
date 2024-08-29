from dataclasses import dataclass, field
from typing import List, Optional, Dict
from copy import deepcopy

from deepeval.test_case import LLMTestCase


@dataclass
class Message:
    llm_test_case: LLMTestCase
    should_evaluate: Optional[bool] = None

    def __post_init__(self):
        # prevent user referencing the wrong LLM test case in a conversation
        self.llm_test_case = deepcopy(self.llm_test_case)


@dataclass
class ConversationalTestCase:
    messages: List[Message]
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    evaluate_all_messages: Optional[bool] = False
    name: Optional[str] = field(default=None)
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.messages) == 0:
            raise TypeError("'messages' must not be empty")

        if not isinstance(self.messages, list) or not all(
            isinstance(item, Message) for item in self.messages
        ):
            raise TypeError("'messages' must be a list of Message")

        for i in range(len(self.messages)):
            message = self.messages[i]
            if self.evaluate_all_messages:
                # Only override should_evaluate if not set
                if message.should_evaluate is None:
                    message.should_evaluate = True
            else:
                # Defaults to only evaluating last message unless overriden
                if (
                    i == len(self.messages) - 1
                    and message.should_evaluate is None
                ):
                    message.should_evaluate = True
                else:
                    if message.should_evaluate is None:
                        message.should_evaluate = False
