from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import random
from ..types import (
    Candidate,
    CandidateId,
    Evaluator,
    GoldenLike,
    ModuleId,
    Prompt,
    PromptRewriter,
    ScoreTable,
)
from ..policies.selection import select_candidate_pareto
from .mutation import NoOpRewriter


@dataclass
class GEPAConfig:
    budget: int
    minibatch_size: int
    seed: int = 0
    # TODO: selection hyperparams go here re-seed every iter
    rewriter: Optional[PromptRewriter] = None


class GEPARunner:
    def __init__(self, evaluator: Evaluator, *, config: GEPAConfig):
        self.evaluator = evaluator
        self.config = config
        self.random_state = random.Random(config.seed)
        self.rewriter = config.rewriter or NoOpRewriter()

        # State
        self.candidates_by_id: Dict[CandidateId, Candidate] = {}
        self.parents: Dict[CandidateId, Optional[CandidateId]] = {}
        self.pareto_scores: ScoreTable = {}

    def _add_candidate(self, cand: Candidate):
        self.candidates_by_id[cand.id] = cand
        self.parents[cand.id] = cand.parent

    def _best_by_mean(self) -> Candidate:
        assert self.pareto_scores, "No scores yet"
        means = {
            cid: sum(vec) / len(vec) for cid, vec in self.pareto_scores.items()
        }
        best_id = max(means, key=means.get)
        return self.candidates_by_id[best_id]

    def optimize(
        self,
        root: Candidate,
        d_feedback: Sequence[GoldenLike],
        d_pareto: Sequence[GoldenLike],
    ) -> Tuple[Candidate, Dict]:
        """
        Returned report dict is a stable place to add lineage, accepted steps, etc.
        """
        # seed pool with root
        self._add_candidate(root)
        # initial Pareto scores for root
        self.pareto_scores[root.id] = self.evaluator.score_on_pareto(root)

        accepted_steps: List[Dict] = []

        remaining = self.config.budget
        while remaining > 0:
            remaining -= 1

            # 1. Pick candidate via Pareto
            k_id = select_candidate_pareto(
                self.pareto_scores, random_state=self.random_state
            )
            parent = self.candidates_by_id[k_id]

            # 2. Pick module
            j: ModuleId = self.evaluator.select_module(parent)

            # 3. Draw minibatch
            b = max(1, min(self.config.minibatch_size, len(d_feedback)))
            if b == 0:
                break
            batch = [
                d_feedback[self.random_state.randrange(0, len(d_feedback))]
                for _ in range(b)
            ]

            # 4. Gather feedback μ_f
            feedback_text = self.evaluator.minibatch_feedback(parent, j, batch)

            # 5. update prompt via rewriter
            old = parent.prompts.get(j, Prompt(text=""))
            new_prompt = self.rewriter.rewrite(
                module_id=j, old_prompt=old, feedback_text=feedback_text
            )

            # 6. Build child cand with prompt swap
            child = Candidate.new(
                prompts=dict(parent.prompts), parent=parent.id
            )
            child.prompts[j] = new_prompt

            # 7. Gate on minibatch improvement
            sigma_before = self.evaluator.minibatch_score(parent, batch)
            sigma_after = self.evaluator.minibatch_score(child, batch)

            if sigma_after > sigma_before:
                # Accept
                self._add_candidate(child)
                self.pareto_scores[child.id] = self.evaluator.score_on_pareto(
                    child
                )
                accepted_steps.append(
                    dict(
                        parent=parent.id,
                        child=child.id,
                        module=j,
                        before=sigma_before,
                        after=sigma_after,
                    )
                )
            # else: reject silently

        best = self._best_by_mean()
        report = dict(
            best_id=best.id,
            accepted_steps=accepted_steps,
            pareto_scores=self.pareto_scores,
            parents=self.parents,
        )
        return best, report
