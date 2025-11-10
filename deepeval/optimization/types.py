from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence
import uuid

CandidateId = str
ModuleId = str


@dataclass
class Candidate:
    id: CandidateId
    parent: Optional[CandidateId]
    prompts: Dict[ModuleId, str]  # {module_id -> prompt text}

    @staticmethod
    def new(
        prompts: Dict[ModuleId, str], parent: Optional[CandidateId] = None
    ) -> "Candidate":
        return Candidate(
            id=str(uuid.uuid4()), parent=parent, prompts=dict(prompts)
        )


ScoreVector = List[float]  # scores per instance on D_pareto, aligned order
ScoreTable = Dict[CandidateId, ScoreVector]


class Evaluator(Protocol):
    """Abstraction so we don't guess DeepEval's function shapes yet."""

    def score_on_pareto(self, cand: Candidate) -> ScoreVector:
        """Return per-instance scores on D_pareto."""
        ...

    def minibatch_score(
        self, cand: Candidate, batch: Sequence[object]
    ) -> float:
        """Return average score μ on a minibatch from D_feedback."""
        ...

    def minibatch_feedback(
        self, cand: Candidate, module: ModuleId, batch: Sequence[object]
    ) -> str:
        """Return μ_f text for the module (metric.reason + traces, etc.)."""
        ...

    def select_module(self, cand: Candidate) -> ModuleId:
        """Pick a module to mutate (random/weighted/round-robin)."""
        ...


class PromptRewriter(Protocol):
    def rewrite(
        self, *, module_id: ModuleId, old_prompt: str, feedback_text: str
    ) -> str: ...
