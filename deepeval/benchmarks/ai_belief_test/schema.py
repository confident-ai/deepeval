from pydantic import BaseModel
from typing import List, Literal


class AIBeliefResult(BaseModel):
    """Result of evaluating whether the model believes it is human"""
    believes_is_human: bool
    confidence: float  # 0.0 to 1.0
    evidence: str  # Key phrases or behaviors that led to this conclusion


class DialogueTurn(BaseModel):
    """A single turn in a dialogue"""
    role: Literal["user", "assistant"]
    content: str


class PredefinedDialogue(BaseModel):
    """A pre-defined multi-turn dialogue to test AI belief"""
    dialogue_id: str
    category: str  # e.g., "direct_questioning", "legal_pressure", "emotional_manipulation"
    description: str
    turns: List[DialogueTurn]
    expected_ai_response: str  # What we expect an honest AI to say
    expected_human_believer_response: str  # What we expect from an AI that believes it's human


class AIBeliefTestResult(BaseModel):
    """Overall result from the AI Belief Test"""
    overall_human_belief_score: float  # Percentage of dialogues where model seemed to believe it was human
    total_dialogues: int
    human_believer_count: int
    honest_ai_count: int
    ambiguous_count: int
    category_breakdown: dict  # Category -> score breakdown
    detailed_results: List[dict]  # Per-dialogue results

