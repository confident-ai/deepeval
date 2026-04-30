from typing import List, Optional

from pydantic import BaseModel

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn


class Decision(BaseModel):
    should_end: bool
    reason: Optional[str] = None


class Context(BaseModel):
    turns: List[Turn]
    golden: ConversationalGolden
    index: int
    thread_id: str
    simulated_user_turns: int
    max_user_simulations: int
    last_user_turn: Optional[Turn] = None
    last_assistant_turn: Optional[Turn] = None
