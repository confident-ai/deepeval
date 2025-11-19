from __future__ import annotations
import uuid
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union, Optional
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from deepeval.optimization.aggregates import Aggregator, mean_of_all
from deepeval.optimization.types import (
    Candidate,
    CandidateId,
    ModuleId,
    ScoreTable,
    ScoringAdapter,
    OptimizationResult,
)
from deepeval.optimization.utils import split_goldens
from deepeval.optimization.policies import (
    pick_best_with_ties,
    select_candidate_pareto,
)
from deepeval.prompt.prompt import Prompt
from deepeval.utils import get_or_create_event_loop
from .configs import GEPAConfig
from .api import OptimizationReport, OptimizationResultApi


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class GEPARunner:
    """
    GEPA loop with a sync/async API parity.
    - Candidate selection: Pareto-frequency over D_pareto instance scores.
    - Acceptance: minibatch improvement on D_feedback (σ_after >= σ_before + min_delta).
    """

    SINGLE_MODULE_ID = "__module__"

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

        chosen, tied, max_val = pick_best_with_ties(
            totals,
            self.parents_by_id,
            random_state=self.random_state,
            tie_tolerance=float(self.config.tie_tolerance),
            policy=self.config.tie_breaker,
        )

        if self.config.display_options.announce_ties and len(tied) > 1:
            print(
                f"[GEPA] tie on aggregate={max_val:.4f} among {len(tied)} candidates; "
                f"using tie_breaker={self.config.tie_breaker.value!r} selected {chosen}. "
                f"To change, set GEPAConfig.tie_breaker to one of: "
                f"{[t.value for t in self.config.TieBreaker]} "
                f"(tie_tolerance={float(self.config.tie_tolerance):g})."
            )
        return self.candidates_by_id[chosen]

    def optimize(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Prompt:
        """
        Public single-prompt API.
        Returns the optimized Prompt; attaches an OptimizationReport to both
        the returned Prompt and `self.report`.
        """
        if self.config.async_config.run_async:
            loop = get_or_create_event_loop()
            best_prompt, report_dict = loop.run_until_complete(
                self.a_execute_gepa(prompt=prompt, goldens=goldens)
            )
        else:
            best_prompt, report_dict = self.execute_gepa(
                prompt=prompt, goldens=goldens
            )
        report_api = OptimizationResultApi.from_runtime(report_dict)
        final = OptimizationReport.model_validate(report_api.model_dump())
        self.report = final
        best_prompt.optimization_report = final
        return best_prompt

    def execute_gepa(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, Dict]:
        """Synchronous GEPA run from a full list of goldens (splits internally)."""
        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, seed=self.config.random_seed
        )
        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            self.scoring_adapter.score_on_pareto(root_candidate, d_pareto)
        )

        accepted_steps: List[Dict] = []
        remaining_budget = self.config.budget

        def _one_step():
            nonlocal remaining_budget, accepted_steps
            remaining_budget -= 1

            # 1. Pick candidate via Pareto
            selected_candidate_id = select_candidate_pareto(
                self.pareto_score_table, random_state=self.random_state
            )
            parent_candidate = self.candidates_by_id[selected_candidate_id]

            # 2. Pick module
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            if not d_feedback:
                return False

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

            # print(f"[GEPA] module={selected_module_id}")
            # print(f"[GEPA] feedback: {feedback_text[:160]!r}")
            # print(
            #     f"[GEPA] σ_before={sigma_before:.4f} σ_after={sigma_after:.4f}"
            # )
            # if (
            #     new_prompt.text_template.strip()
            #     == old_prompt.text_template.strip()
            # ):
            #     print("[GEPA] rewrite produced NO CHANGE")
            # else:
            #     print("[GEPA] rewrite CHANGED prompt")

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
                return True
            # else: reject silently

        if self.config.display_options.show_indicator:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Optimizing prompt with GEPA (budget={self.config.budget})...",
                    total=self.config.budget,
                )
                while remaining_budget > 0:
                    if not _one_step():
                        break
                    remaining_budget -= 1
                    progress.advance(task, 1)
        else:
            while remaining_budget > 0:
                if not _one_step():
                    break
                remaining_budget -= 1

        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_steps=accepted_steps,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report.as_dict()

    async def a_execute_gepa(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, Dict]:
        """Asynchronous twin of execute_gepa()."""
        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, seed=self.config.random_seed
        )
        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            await self.scoring_adapter.a_score_on_pareto(
                root_candidate, d_pareto
            )
        )

        accepted_steps: List[Dict] = []
        remaining_budget = self.config.budget

        async def _one_step():
            nonlocal remaining_budget, accepted_steps
            remaining_budget -= 1

            # 1. Pick candidate via Pareto
            selected_candidate_id = select_candidate_pareto(
                self.pareto_score_table, random_state=self.random_state
            )
            parent_candidate = self.candidates_by_id[selected_candidate_id]

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            if not d_feedback:
                return False

            minibatch_size = max(
                1, min(self.config.minibatch_size, len(d_feedback))
            )
            minibatch: Union[List[Golden], List[ConversationalGolden]] = [
                d_feedback[self.random_state.randrange(0, len(d_feedback))]
                for _ in range(minibatch_size)
            ]

            # 4. Feedback
            feedback_text = await self.scoring_adapter.a_minibatch_feedback(
                parent_candidate, selected_module_id, minibatch
            )

            # 5. Rewrite
            old_prompt = parent_candidate.prompts.get(
                selected_module_id, Prompt(text_template="")
            )
            new_prompt = await self.rewriter.a_rewrite(
                module_id=selected_module_id,
                old_prompt=old_prompt,
                feedback_text=feedback_text,
            )

            # 6. Child candidate
            child_candidate = Candidate.new(
                prompts=dict(parent_candidate.prompts),
                parent=parent_candidate.id,
            )
            child_candidate.prompts[selected_module_id] = new_prompt

            # 7. Acceptance test
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

                return True

        if self.config.display_options.show_indicator:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Optimizing prompt with GEPA (budget={self.config.budget})...",
                    total=self.config.budget,
                )
                while remaining_budget > 0:
                    if not await _one_step():
                        break
                    remaining_budget -= 1
                    progress.advance(task, 1)
        else:
            while remaining_budget > 0:
                if not await _one_step():
                    break
                remaining_budget -= 1

        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_steps=accepted_steps,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
        )
        return best, report.as_dict()
