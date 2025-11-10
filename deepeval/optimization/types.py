from __future__ import annotations
import uuid

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
)


CandidateId = str
ModuleId = str


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


GoldenLike = TypeVar("GoldenLike", "Golden", "ConversationalGolden")
CandidateId = str
ModuleId = str
ScoreVector = List[float]  # scores per instance on D_pareto, aligned order
ScoreTable = Dict[CandidateId, ScoreVector]


@dataclass
class Candidate:
    id: CandidateId
    parent: Optional[CandidateId]
    prompts: Dict[ModuleId, Prompt]

    @staticmethod
    def new(
        prompts: Dict[ModuleId, Prompt],
        parent: Optional[CandidateId] = None,
    ) -> "Candidate":
        return Candidate(
            id=str(uuid.uuid4()), parent=parent, prompts=dict(prompts)
        )


@dataclass
class Prompt:
    """
    text: used for single text prompts
    messages: used for chat style prompts
    model_settings: allows for extensibility without breaking interface
    """

    text: str
    messages: Optional[List[Dict[str, str]]] = None
    model_settings: Dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    """Abstraction so we don't guess DeepEval's function shapes yet."""

    def score_on_pareto(self, cand: Candidate) -> ScoreVector:
        """Return per-instance scores on D_pareto."""
        ...

    def minibatch_score(
        self, cand: Candidate, batch: Sequence[GoldenLike]
    ) -> float:
        """Return average score μ on a minibatch from D_feedback."""
        ...

    def minibatch_feedback(
        self, cand: Candidate, module: ModuleId, batch: Sequence[GoldenLike]
    ) -> str:
        """Return μ_f text for the module (metric.reason + traces, etc.)."""
        ...

    def select_module(self, cand: Candidate) -> ModuleId:
        """Pick a module to mutate (random/weighted/round-robin)."""
        ...


class PromptRewriter(Protocol):
    def rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt: ...
