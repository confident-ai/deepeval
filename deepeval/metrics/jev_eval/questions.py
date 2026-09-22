"""The three decision points a JevEval metric is built from.

Each question is answered by Jev with calibrated probabilities, never text,
and each has a fixed mapping onto a value in ``[0, 1]`` (see ``utils.py``):

- ``Noul``: one proposition about the test case, ``v = P(true)``.
- ``Score``: one holistic judgement over ordered descriptive levels,
  ``v = expected level index / (levels - 1)``.
- ``Choice``: one selection from an unordered closed set where every option
  carries a *credit* in ``[0, 1]``, or ``None`` when that option means the
  question does not apply to this test case.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from deepeval.models.system_one.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneQuestion,
)

# Jev accepts at most this many ordered levels in one Score question.
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


class _Question(BaseModel):
    weight: float = Field(default=1.0, gt=0)

    @property
    def text(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_system_one(self) -> SystemOneQuestion:  # pragma: no cover
        raise NotImplementedError


class Noul(_Question):
    """A proposition Jev judges true or false about the state."""

    type: Literal["noul"] = "noul"
    statement: str
    true: Optional[Any] = None
    false: Optional[Any] = None

    def __init__(self, statement: Optional[str] = None, **data: Any):
        if statement is not None:
            data["statement"] = statement
        super().__init__(**data)

    @field_validator("statement")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Noul statement cannot be empty.")
        return value

    @property
    def text(self) -> str:
        return self.statement

    def to_system_one(self) -> NoulQuestion:
        return NoulQuestion(
            instructions=self.statement, true=self.true, false=self.false
        )


class Score(_Question):
    """A holistic judgement over ordered descriptive levels, worst first."""

    type: Literal["score"] = "score"
    question: str
    levels: List[str]

    def __init__(self, question: Optional[str] = None, **data: Any):
        if question is not None:
            data["question"] = question
        super().__init__(**data)

    @field_validator("question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Score question cannot be empty.")
        return value

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, levels: List[str]) -> List[str]:
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"Score needs between {MIN_SCORE_LEVELS} and "
                f"{MAX_SCORE_LEVELS} levels; got {len(levels)}."
            )
        if any(not level or not str(level).strip() for level in levels):
            raise ValueError("Score levels cannot be empty.")
        if len(set(levels)) != len(levels):
            raise ValueError("Score levels must be unique.")
        return levels

    @property
    def text(self) -> str:
        return self.question

    def to_system_one(self) -> ScoreQuestion:
        return ScoreQuestion(instructions=self.question, levels=self.levels)


class Choice(_Question):
    """A selection from a closed set. ``options`` maps each option name to the
    credit it earns in ``[0, 1]``; ``None`` marks an option that means the
    question does not apply to this test case."""

    type: Literal["choice"] = "choice"
    question: str
    options: Dict[str, Optional[float]]

    def __init__(self, question: Optional[str] = None, **data: Any):
        if question is not None:
            data["question"] = question
        super().__init__(**data)

    @field_validator("question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Choice question cannot be empty.")
        return value

    @field_validator("options")
    @classmethod
    def _check_options(
        cls, options: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        if len(options) < 2:
            raise ValueError("Choice needs at least 2 options.")
        if any(not name or not name.strip() for name in options):
            raise ValueError("Choice option names cannot be empty.")
        for name, credit in options.items():
            if credit is None:
                continue
            if not 0.0 <= float(credit) <= 1.0:
                raise ValueError(
                    f"Choice option '{name}' has credit {credit}; credits "
                    "must be between 0 and 1, or None for not applicable."
                )
        if all(credit is None for credit in options.values()):
            raise ValueError(
                "Choice needs at least one option with a credit; every "
                "option is None (not applicable)."
            )
        return options

    @property
    def text(self) -> str:
        return self.question

    @property
    def applicable_options(self) -> Dict[str, float]:
        return {
            name: float(credit)
            for name, credit in self.options.items()
            if credit is not None
        }

    @property
    def not_applicable_options(self) -> List[str]:
        return [name for name, credit in self.options.items() if credit is None]

    def to_system_one(self) -> ChoiceQuestion:
        # Credits stay on the DeepEval side; Jev only sees the option names.
        return ChoiceQuestion(
            instructions=self.question,
            options={name: None for name in self.options},
        )


JevQuestion = Union[Noul, Score, Choice]


class QuestionOutcome(BaseModel):
    """One entry of ``metric.score_breakdown``."""

    question: str
    type: Literal["noul", "score", "choice"]
    weight: float
    value: Optional[float]
    applicable: bool
    probabilities: Dict[str, float]
    confidence: Optional[float] = None
