from __future__ import annotations
import uuid
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union, Optional

from deepeval.optimization.aggregates import Aggregator, mean_of_all
from deepeval.optimization.types import (
    Candidate,
    CandidateId,
    ModuleId,
    ScoreTable,
    ScoringAdapter,
    OptimizationResult,
)
from deepeval.optimization.utils import normalize_seed_prompts, split_goldens
from deepeval.optimization.policies.selection import select_candidate_pareto
from deepeval.prompt.prompt import Prompt
from .configs import GEPAConfig


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class GEPARunner:
    """
    GEPA loop with a sync/async API parity.
    - Candidate selection: Pareto-frequency over D_pareto instance scores.
    - Acceptance: minibatch improvement on D_feedback (σ_after >= σ_before + min_delta).
    """

    def __init__(
        self,
        *,
        scoring_adapter: ScoringAdapter,
        config: GEPAConfig,
        aggregate_instances: Aggregator = mean_of_all,
    ):
        self.optimization_id: str = str(uuid.uuid4())
        self.scoring_adapter = scoring_adapter
        self.config = config
        self.random_state = random.Random(config.random_seed)
        self.rewriter = config.get_rewriter()
        self.aggregate_instances = aggregate_instances

        # State
        self.candidates_by_id: Dict[CandidateId, Candidate] = {}
        self.parents_by_id: Dict[CandidateId, Optional[CandidateId]] = {}
        self.pareto_score_table: ScoreTable = {}

    def _add_candidate(self, candidate: Candidate):
        self.candidates_by_id[candidate.id] = candidate
        self.parents_by_id[candidate.id] = candidate.parent

    def _best_by_aggregate(self) -> Candidate:
        assert self.pareto_score_table, "No scores yet"
        totals = {
            candidate_id: self.aggregate_instances(vector)
            for candidate_id, vector in self.pareto_score_table.items()
        }
        best_candidate_id = max(totals, key=totals.get)
        return self.candidates_by_id[best_candidate_id]

    def optimize(
        self,
        *,
        seed_prompts: Union[Dict[ModuleId, Prompt], List[Prompt]],
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Candidate, Dict]:
        """Synchronous GEPA run from a full list of goldens (splits internally)."""
        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, seed=self.config.random_seed
        )
        seed_prompts_by_module = normalize_seed_prompts(seed_prompts)
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            self.scoring_adapter.score_on_pareto(root_candidate, d_pareto)
        )

        accepted_steps: List[Dict] = []

        remaining_budget = self.config.budget
        while remaining_budget > 0:
            remaining_budget -= 1

            # 1. Pick candidate via Pareto
            selected_candidate_id = select_candidate_pareto(
                self.pareto_score_table, random_state=self.random_state
            )
            parent_candidate = self.candidates_by_id[selected_candidate_id]

            # 2. Pick module
            selected_module_id: ModuleId = self.scoring_adapter.select_module(
                parent_candidate
            )

            # 3. Draw minibatch
            if not d_feedback:
                break

            minibatch_size = max(
                1, min(self.config.minibatch_size, len(d_feedback))
            )

            minibatch: Union[List[Golden], List[ConversationalGolden]] = [
                d_feedback[self.random_state.randrange(0, len(d_feedback))]
                for _ in range(minibatch_size)
            ]

            # 4. Gather feedback μ_f
            feedback_text = self.scoring_adapter.minibatch_feedback(
                parent_candidate, selected_module_id, minibatch
            )

            # 5. update prompt via rewriter
            old_prompt = parent_candidate.prompts.get(
                selected_module_id, Prompt(text_template="")
            )
            new_prompt = self.rewriter.rewrite(
                module_id=selected_module_id,
                old_prompt=old_prompt,
                feedback_text=feedback_text,
            )

            # 6. Build child cand with prompt swap
            child_candidate = Candidate.new(
                prompts=dict(parent_candidate.prompts),
                parent=parent_candidate.id,
            )
            child_candidate.prompts[selected_module_id] = new_prompt

            # 7. Gate on minibatch improvement
            sigma_before = self.scoring_adapter.minibatch_score(
                parent_candidate, minibatch
            )
            sigma_after = self.scoring_adapter.minibatch_score(
                child_candidate, minibatch
            )

            if sigma_after >= sigma_before + self.config.min_delta:
                # Accept
                self._add_candidate(child_candidate)
                self.pareto_score_table[child_candidate.id] = (
                    self.scoring_adapter.score_on_pareto(
                        child_candidate, d_pareto
                    )
                )
                accepted_steps.append(
                    dict(
                        parent=parent_candidate.id,
                        child=child_candidate.id,
                        module=selected_module_id,
                        before=sigma_before,
                        after=sigma_after,
                    )
                )
            # else: reject silently

        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_steps=accepted_steps,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
        )
        return best, report.as_dict()

    async def a_optimize(
        self,
        *,
        seed_prompts: Union[Dict[ModuleId, Prompt], List[Prompt]],
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Candidate, Dict]:
        """Asynchronous twin of optimize()."""
        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, seed=self.config.random_seed
        )
        seed_prompts_by_module = normalize_seed_prompts(seed_prompts)
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            await self.scoring_adapter.a_score_on_pareto(
                root_candidate, d_pareto
            )
        )

        accepted_steps: List[Dict] = []
        remaining_budget = self.config.budget
        while remaining_budget > 0:
            remaining_budget -= 1
            selected_candidate_id = select_candidate_pareto(
                self.pareto_score_table, random_state=self.random_state
            )
            parent_candidate = self.candidates_by_id[selected_candidate_id]
            selected_module_id: ModuleId = (
                await self.scoring_adapter.a_select_module(parent_candidate)
            )

            if not d_feedback:
                break

            minibatch_size = max(
                1, min(self.config.minibatch_size, len(d_feedback))
            )

            minibatch: Union[List[Golden], List[ConversationalGolden]] = [
                d_feedback[self.random_state.randrange(0, len(d_feedback))]
                for _ in range(minibatch_size)
            ]

            feedback_text = await self.scoring_adapter.a_minibatch_feedback(
                parent_candidate, selected_module_id, minibatch
            )
            old_prompt = parent_candidate.prompts.get(
                selected_module_id, Prompt(text_template="")
            )
            new_prompt = await self.rewriter.a_rewrite(
                module_id=selected_module_id,
                old_prompt=old_prompt,
                feedback_text=feedback_text,
            )

            child_candidate = Candidate.new(
                prompts=dict(parent_candidate.prompts),
                parent=parent_candidate.id,
            )
            child_candidate.prompts[selected_module_id] = new_prompt

            sigma_before = await self.scoring_adapter.a_minibatch_score(
                parent_candidate, minibatch
            )
            sigma_after = await self.scoring_adapter.a_minibatch_score(
                child_candidate, minibatch
            )
            if sigma_after >= sigma_before + self.config.min_delta:
                self._add_candidate(child_candidate)
                self.pareto_score_table[child_candidate.id] = (
                    await self.scoring_adapter.a_score_on_pareto(
                        child_candidate, d_pareto
                    )
                )
                accepted_steps.append(
                    dict(
                        parent=parent_candidate.id,
                        child=child_candidate.id,
                        module=selected_module_id,
                        before=sigma_before,
                        after=sigma_after,
                    )
                )

        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_steps=accepted_steps,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
        )
        return best, report.as_dict()
