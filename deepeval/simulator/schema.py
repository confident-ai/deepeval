from typing import Optional

from pydantic import BaseModel


class ConversationCompletion(BaseModel):
    is_complete: bool
    reason: str


class SimulatedInput(BaseModel):
    simulated_input: str


class EdgeChoice(BaseModel):
    """Result of LLM edge classification for a `SimulationNode`.

    `index` is the 1-based position of the matching outgoing edge, or `None`
    when the assistant reply did not match any edge ("none of the above" —
    the runner stays on the current node in that case).
    """

    index: Optional[int] = None
    reason: Optional[str] = None


class InterruptDecision(BaseModel):
    """LLM mid-speech decision for whether the simulated user should barge in.

    When `should_interrupt` is True, `utterance` must be the barge-in line the
    user will speak over the agent.
    """

    should_interrupt: bool
    reason: str
    utterance: Optional[str] = None
