from __future__ import annotations
import uuid
import random
import time

from contextlib import contextmanager
from pydantic import BaseModel as PydanticBaseModel
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Tuple,
    TYPE_CHECKING,
    Union,
    Optional,
)
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from deepeval.errors import DeepEvalError
from deepeval.evaluate.configs import AsyncConfig
from deepeval.optimization.aggregates import Aggregator, mean_of_all
from deepeval.optimization.types import (
    AcceptedIterationDict,
    Candidate,
    CandidateId,
    ModuleId,
    ScoreTable,
    ScoringAdapter,
    OptimizationResult,
)
from deepeval.optimization.adapters.deepeval_scoring_adapter import (
    DeepEvalScoringAdapter,
)
from deepeval.optimization.utils import split_goldens
from deepeval.optimization.policies import (
    pick_best_with_ties,
    select_candidate_pareto,
)
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.prompt.prompt import Prompt
from deepeval.utils import get_or_create_event_loop
from .configs import GEPAConfig
from deepeval.optimization.types import OptimizationReport


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
        config: GEPAConfig,
        async_config: Optional[AsyncConfig] = AsyncConfig(),
        aggregate_instances: Aggregator = mean_of_all,
        scoring_adapter: Optional[ScoringAdapter] = None,
        # used if scoring_adapter is None
        metrics: Optional[
            Union[List[BaseMetric], List[BaseConversationalMetric]]
        ] = None,
        model: DeepEvalBaseLLM,
        model_schema: Optional[PydanticBaseModel] = None,
    ):
        self.optimization_id: str = str(uuid.uuid4())
        self.config = config
        self.async_config = async_config
        self.random_state = random.Random(config.random_seed)
        self.model = model
        self.scoring_adapter = scoring_adapter
        self.rewriter = config.get_rewriter()
        self.aggregate_instances = aggregate_instances

        if scoring_adapter is None:
            if not metrics:
                raise DeepEvalError(
                    "Provide `metrics` when no `scoring_adapter` is supplied."
                )
            self.scoring_adapter = DeepEvalScoringAdapter(
                metrics=metrics, model=model, model_schema=model_schema
            )

        if hasattr(self.scoring_adapter, "configure_async"):
            self.scoring_adapter.configure_async(
                max_concurrent=self.async_config.max_concurrent,
                throttle_seconds=float(self.async_config.throttle_value),
            )

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
        Returns the optimized Prompt. Attaches an OptimizationReport to both
        the returned Prompt and `self.report`.
        """
        if self.async_config.run_async:
            loop = get_or_create_event_loop()
            best_prompt, report_dict = loop.run_until_complete(
                self.a_execute_gepa(prompt=prompt, goldens=goldens)
            )
        else:
            best_prompt, report_dict = self.execute_gepa(
                prompt=prompt, goldens=goldens
            )

        self.report = OptimizationReport.from_runtime(report_dict)
        best_prompt.optimization_report = self.report
        return best_prompt

    def execute_gepa(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, Dict]:
        """Synchronous GEPA run from a full list of goldens (splits internally)."""
        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, random_state=self.random_state
        )
        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            self.scoring_adapter.score_on_pareto(root_candidate, d_pareto)
        )

        accepted_iterations: List[Dict] = []
        remaining_iterations = self.config.iterations

        def _one_iteration():
            nonlocal remaining_iterations, accepted_iterations

            if not d_feedback:
                return False

            # 1. Pick candidate via Pareto
            parent_candidate = self._pick_candidate()

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            minibatch = self._draw_minibatch(d_feedback)

            # 4. Feedback
            feedback_text = self.scoring_adapter.minibatch_feedback(
                parent_candidate, selected_module_id, minibatch
            )

            child_prompt = self._generate_child_prompt(
                selected_module_id, parent_candidate, feedback_text
            )
            if child_prompt is None:
                # child prompt matched parent. Skip this generation.
                return True

            child_candidate = self._make_child(
                selected_module_id, parent_candidate, child_prompt
            )

            parent_score = self.scoring_adapter.minibatch_score(
                parent_candidate, minibatch
            )

            child_score = self.scoring_adapter.minibatch_score(
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

            # 7. Acceptance test
            if self._should_accept_child(parent_score, child_score):
                accepted_iterations.append(
                    self._accept_child(
                        selected_module_id,
                        parent_candidate,
                        child_candidate,
                        d_pareto,
                        parent_score,
                        child_score,
                    )
                )

            return True

        self._run_loop_iteration(_one_iteration)
        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
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
            goldens, self.config.pareto_size, random_state=self.random_state
        )
        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_candidate = Candidate.new(prompts=dict(seed_prompts_by_module))
        self._add_candidate(root_candidate)
        self.pareto_score_table[root_candidate.id] = (
            await self.scoring_adapter.a_score_on_pareto(
                root_candidate, d_pareto
            )
        )

        accepted_iterations: List[Dict] = []
        remaining_iterations = self.config.iterations

        async def _one_iteration():
            nonlocal remaining_iterations, accepted_iterations

            if not d_feedback:
                return False

            # 1. Pick candidate via Pareto
            parent_candidate = self._pick_candidate()

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            minibatch = self._draw_minibatch(d_feedback)

            # 4. Feedback
            feedback_text = await self.scoring_adapter.a_minibatch_feedback(
                parent_candidate, selected_module_id, minibatch
            )

            child_prompt = await self._a_generate_child_prompt(
                selected_module_id, parent_candidate, feedback_text
            )
            if child_prompt is None:
                # child prompt matched parent. Skip this generation.
                return True

            child_candidate = self._make_child(
                selected_module_id, parent_candidate, child_prompt
            )

            parent_score = await self.scoring_adapter.a_minibatch_score(
                parent_candidate, minibatch
            )

            child_score = await self.scoring_adapter.a_minibatch_score(
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

            # 7. Acceptance test
            if self._should_accept_child(parent_score, child_score):
                accepted_iterations.append(
                    await self._a_accept_child(
                        selected_module_id,
                        parent_candidate,
                        child_candidate,
                        d_pareto,
                        parent_score,
                        child_score,
                    )
                )

            return True

        await self._a_run_loop_iteration(_one_iteration)
        best = self._best_by_aggregate()
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report.as_dict()

    def _pick_candidate(self) -> Candidate:

        selected_candidate_id = select_candidate_pareto(
            self.pareto_score_table, random_state=self.random_state
        )
        return self.candidates_by_id[selected_candidate_id]

    def _draw_minibatch(
        self, d_feedback: Union[List[Golden], List[ConversationalGolden]]
    ) -> Union[List[Golden], List[ConversationalGolden]]:
        minibatch_size = max(
            1, min(self.config.minibatch_size, len(d_feedback))
        )
        return [
            d_feedback[self.random_state.randrange(0, len(d_feedback))]
            for _ in range(minibatch_size)
        ]

    async def _a_generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_candidate: Candidate,
        feedback_text: str,
    ) -> Optional[Prompt]:
        # 5. Rewrite
        old_prompt = parent_candidate.prompts.get(
            selected_module_id, Prompt(text_template="")
        )
        new_prompt = await self.rewriter.a_rewrite(
            model=self.model,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )

        old_txt = old_prompt.text_template.strip()
        new_txt = new_prompt.text_template.strip()
        if new_txt == old_txt:
            # don't accept if new prompt is the same as parent
            return None
        return new_prompt

    def _generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_candidate: Candidate,
        feedback_text: str,
    ):
        # 5. Rewrite
        old_prompt = parent_candidate.prompts.get(
            selected_module_id, Prompt(text_template="")
        )
        new_prompt = self.rewriter.rewrite(
            model=self.model,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )

        old_txt = old_prompt.text_template.strip()
        new_txt = new_prompt.text_template.strip()
        if new_txt == old_txt:
            # don't accept if new prompt is the same as parent
            return None
        return new_prompt

    def _make_child(
        self,
        selected_module_id: ModuleId,
        parent_candidate: Candidate,
        child_prompt: Prompt,
    ) -> Candidate:

        # 6. Child candidate
        child_candidate = Candidate.new(
            prompts=dict(parent_candidate.prompts),
            parent=parent_candidate.id,
        )
        child_candidate.prompts[selected_module_id] = child_prompt
        return child_candidate

    def _should_accept_child(
        self, parent_score: float, child_score: float
    ) -> bool:
        jitter = 1e-6
        return child_score >= parent_score + max(self.config.min_delta, jitter)

    def _accept_child(
        self,
        selected_module_id: ModuleId,
        parent_candidate: Candidate,
        child_candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
        parent_score: float,
        child_score: float,
    ) -> AcceptedIterationDict:
        self._add_candidate(child_candidate)
        self.pareto_score_table[child_candidate.id] = (
            self.scoring_adapter.score_on_pareto(child_candidate, d_pareto)
        )

        return AcceptedIterationDict(
            parent=parent_candidate.id,
            child=child_candidate.id,
            module=selected_module_id,
            before=parent_score,
            after=child_score,
        )

    async def _a_accept_child(
        self,
        selected_module_id: ModuleId,
        parent_candidate: Candidate,
        child_candidate: Candidate,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
        parent_score: float,
        child_score: float,
    ) -> AcceptedIterationDict:
        self._add_candidate(child_candidate)
        self.pareto_score_table[child_candidate.id] = (
            await self.scoring_adapter.a_score_on_pareto(
                child_candidate, d_pareto
            )
        )

        return AcceptedIterationDict(
            parent=parent_candidate.id,
            child=child_candidate.id,
            module=selected_module_id,
            before=parent_score,
            after=child_score,
        )

    def _format_progress_description(
        self, iteration: int, remaining_iterations: int, elapsed_time: float
    ) -> str:
        return (
            f"Optimizing prompt with GEPA (iterations={self.config.iterations}) "
            f"[rgb(25,227,160)]• Iteration {iteration}/{self.config.iterations} "
            f"• {elapsed_time:.2f}s • remaining={remaining_iterations}"
        )

    def _progress_columns(self):
        return (
            SpinnerColumn(style="rgb(106,0,255)"),
            BarColumn(bar_width=60),
            TextColumn("[progress.description]{task.description}"),
        )

    @contextmanager
    def _maybe_progress(self) -> Progress:
        """Context manager yielding a Progress or a no-op"""
        if self.config.display_options.show_indicator:
            with Progress(*self._progress_columns(), transient=True) as p:
                yield p
        else:

            class _Noop:
                def add_task(self, *_, **__):
                    return 0

                def advance(self, *_, **__):
                    pass

                def update(self, *_, **__):
                    pass

            yield _Noop()

    def _run_loop_iteration(
        self,
        gepa_iteration: Callable[[], bool],
    ) -> None:
        remaining = self.config.iterations
        iteration = 0
        with self._maybe_progress() as progress:
            task = progress.add_task(
                f"Optimizing prompt with GEPA (iterations={remaining})...",
                total=remaining,
            )
            while remaining > 0:
                iteration += 1
                start_time = time.perf_counter()
                ok = gepa_iteration()
                end_time = time.perf_counter()
                if not ok:
                    break
                remaining -= 1
                progress.advance(task, 1)
                progress.update(
                    task,
                    description=self._format_progress_description(
                        iteration, remaining, end_time - start_time
                    ),
                )

    async def _a_run_loop_iteration(
        self,
        a_gepa_iteration: Callable[[], Awaitable[bool]],
    ) -> None:
        remaining = self.config.iterations
        iteration = 0
        with self._maybe_progress() as progress:
            task = progress.add_task(
                f"Optimizing prompt with GEPA (iterations={remaining})...",
                total=remaining,
            )
            while remaining > 0:
                iteration += 1
                start_time = time.perf_counter()
                ok = await a_gepa_iteration()
                end_time = time.perf_counter()
                if not ok:
                    break
                remaining -= 1
                progress.advance(task, 1)
                progress.update(
                    task,
                    description=self._format_progress_description(
                        iteration, remaining, end_time - start_time
                    ),
                )
