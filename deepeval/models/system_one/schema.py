from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class NoulQuestion(BaseModel):
    instructions: Any
    true: Optional[Any] = None
    false: Optional[Any] = None


class ChoiceQuestion(BaseModel):
    instructions: Any
    options: Dict[str, Any]


class ScoreQuestion(BaseModel):
    instructions: Any
    levels: List[Any]


SystemOneQuestion = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class NoulAnswer(BaseModel):
    probability: float

    @property
    def confidence(self) -> float:
        """How decisive the answer is, on the same scale as the confidence the
        API returns for Choice and Score answers.

        A Noul answer carries no confidence of its own because ``P(no)`` is
        already ``1 - P(yes)``. TypeSafe derives Choice/Score confidence from
        the probability distribution as ``(n * peak - 1) / (n - 1)``; with the
        two outcomes of a Noul that reduces to ``|2p - 1|``: 0 at ``p = 0.5``,
        1 at either extreme, symmetric between yes and no."""
        return abs(2.0 * self.probability - 1.0)


class ChoiceAnswer(BaseModel):
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    score: float
    probabilities: Dict[int, float]
    confidence: float

    @property
    def normalized(self) -> float:
        top = max(self.probabilities) if self.probabilities else 0
        return self.score / top if top > 0 else 0.0


class SystemOneAnswers(BaseModel):
    nouls: Dict[str, NoulAnswer] = Field(default_factory=dict)
    choices: Dict[str, ChoiceAnswer] = Field(default_factory=dict)
    scores: Dict[str, ScoreAnswer] = Field(default_factory=dict)

    def min_confidence(self) -> Optional[float]:
        """The least decisive answer in the request, or ``None`` when there
        are no answers. One question near the fence is enough to make an
        aggregate over the request untrustworthy, so the minimum rather than
        the mean is what a confidence gate should look at."""
        values = [a.confidence for a in self.nouls.values()]
        values += [a.confidence for a in self.choices.values()]
        values += [a.confidence for a in self.scores.values()]
        return min(values) if values else None
