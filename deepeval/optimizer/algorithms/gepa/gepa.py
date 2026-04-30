from __future__ import annotations
import uuid
import random
import time
from rich.table import Table
from rich import box
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

from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval.errors import DeepEvalError
from deepeval.optimizer.scorer.schema import ScorerDiagnosisResult
from deepeval.optimizer.utils import Aggregator, mean_of_all
from deepeval.optimizer.types import (
    AcceptedIterationDict,
    PromptConfiguration,
    PromptConfigurationId,
    ModuleId,
    ScoreTable,
    OptimizationReport,
    RunnerStatusType,
    RunnerStatusCallback,
)
from deepeval.optimizer.scorer.base import BaseScorer
from deepeval.optimizer.algorithms.base import BaseAlgorithm
from deepeval.optimizer.utils import (
    split_goldens,
    build_prompt_config_snapshots,
)
from deepeval.optimizer.policies import (
    pick_best_with_ties,
    select_prompt_configuration_pareto,
    _is_dominated
)
from deepeval.prompt.api import PromptType
from deepeval.prompt.prompt import Prompt
from deepeval.optimizer.rewriter import Rewriter
from deepeval.optimizer.policies import TieBreaker
from deepeval.optimizer.algorithms.configs import (
    GEPA_MIN_DELTA,
    GEPA_TIE_TOLERANCE,
    GEPA_REWRITE_INSTRUCTION_MAX_CHARS,
)

if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class GEPA(BaseAlgorithm):
    """
    GEPA loop with sync/async execution.

    This runner is intentionally low level and does not know about metrics,
    models, or async configs. It relies on a preconfigured
    Scorer and Rewriter, which are typically constructed by
    the higher-level PromptOptimizer.

    Parameters
    ----------
    iterations : int
        Total number of GEPA loop iterations (mutation attempts). Default is 5.
    minibatch_size : int
        Number of examples drawn from D_feedback per iteration. Default is 8.
    pareto_size : int
        Size of the Pareto validation subset D_pareto. Default is 3.
    patience : int
        If there's no improvement in the Pareto score table for the last patience iterations, stop the optimization. Default is 3.
    random_seed : int, optional
        RNG seed for reproducibility. If None, derived from time.time_ns().
    tie_breaker : TieBreaker
        Policy for breaking ties. Default is TieBreaker.PREFER_CHILD.
    """

    name = "GEPA"
    SINGLE_MODULE_ID: ModuleId = "__module__"
    TieBreaker = TieBreaker

    def __init__(
        self,
        iterations: int = 5,
        minibatch_size: int = 8,
        pareto_size: int = 3,
        random_seed: Optional[int] = None,
        patience: int = 3,
        tie_breaker: TieBreaker = TieBreaker.PREFER_CHILD,
        aggregate_instances: Aggregator = mean_of_all,
        reflection_model: Optional[DeepEvalBaseLLM] = "gpt-4o-mini",
        mutation_model: Optional[DeepEvalBaseLLM] = "gpt-4o",
        scorer: Optional[BaseScorer] = None,
    ) -> None:
        # Validate parameters
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")
        if pareto_size < 1:
            raise ValueError("pareto_size must be >= 1")

        self.iterations = iterations
        self.minibatch_size = minibatch_size
        self.pareto_size = pareto_size
        self.patience = patience
        self.tie_breaker = tie_breaker
        self.aggregate_instances = aggregate_instances
        self.scorer = scorer

        # If no seed provided, use time-based seed
        if random_seed is None:
            random_seed = time.time_ns()
        self.random_seed = random_seed
        self.random_state = random.Random(random_seed)

        # runtime state to be reset between runs
        self.reset_state()

        # Status callback set by PromptOptimizer:
        #   (kind, step_index, total_steps, detail) -> None
        self.status_callback: Optional[RunnerStatusCallback] = None
        self.step_callback: Optional[Callable[[str], None]] = None

        self.reflection_model: Optional["DeepEvalBaseLLM"] = reflection_model
        self.mutation_model: Optional["DeepEvalBaseLLM"] = mutation_model

        # lazy loaded
        self._rewriter: Optional[Rewriter] = None

    ##############
    # Public API #
    ##############

    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, OptimizationReport]:
        """Synchronous GEPA run from a full list of goldens (splits internally)."""
        total_goldens = len(goldens)
        if total_goldens < 2:
            raise DeepEvalError(
                "GEPA prompt optimization requires at least 2 goldens, but "
                f"received {total_goldens}. Provide at least two goldens to "
                "run the optimizer."
            )

        if self.reflection_model is not None:
            self.scorer.optimizer_model = self.reflection_model
        if self.mutation_model is not None:
            self._rewriter.optimizer_model = self.mutation_model

        self._ensure_scorer()
        self.reset_state()

        d_feedback, d_pareto = split_goldens(
            goldens, self.pareto_size, random_state=self.random_state
        )

        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []
        consecutive_rejections = 0

        def _one_iteration() -> bool:
            nonlocal accepted_iterations
            nonlocal consecutive_rejections
            
            if not d_feedback:
                return False

            iter_start = time.perf_counter()

            # Seed Pareto scores lazily on first iteration
            if not self.pareto_score_table:
                self.pareto_score_table[root_prompt_configuration.id] = (
                    self.scorer.score_pareto(
                        root_prompt_configuration, d_pareto
                    )
                )

            # 1. Pick prompt_configuration via Pareto
            parent_prompt_configuration = self._pick_prompt_configuration()

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            minibatch = self._draw_minibatch(d_feedback)

            # 4. Feedback
            feedback_diagnosis = self.scorer.get_minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            parent_minibatch_score = self.scorer.score_minibatch(
                parent_prompt_configuration, minibatch
            )

            # 5. Rewrite
            child_prompt = self._generate_child_prompt(
                selected_module_id, parent_prompt_configuration, feedback_diagnosis
            )
            if child_prompt is None:
                # Child prompt matched parent; skip this iteration.
                return True

            # 6. Child prompt_configuration
            child_prompt_configuration = self._make_child(
                selected_module_id, parent_prompt_configuration, child_prompt
            )

            child_minibatch_score = self.scorer.score_minibatch(
                child_prompt_configuration, minibatch
            )

            if child_minibatch_score <= parent_minibatch_score:
                parent_agg = self.aggregate_instances(
                    self.pareto_score_table[parent_prompt_configuration.id]
                )
                self._iteration_log.append({
                    "iteration": self._current_iteration,
                    "outcome": "skipped",
                    "reason": f"Skipped (minibatch score did not improve)",
                    "before": parent_agg,
                    "after": child_minibatch_score,
                    "elapsed": time.perf_counter() - iter_start,
                })
                return True

            # 7. Evaluate child on the GLOBAL validation set (d_pareto)
            child_pareto_scores = self.scorer.score_pareto(
                child_prompt_configuration, d_pareto
            )
            parent_pareto_scores = self.pareto_score_table[parent_prompt_configuration.id]

            # 8. Acceptance test (Pareto non-domination vs parent)
            accepted = self._should_accept_child(child_pareto_scores, parent_pareto_scores)

            if accepted:
                consecutive_rejections = 0
                parent_agg = self.aggregate_instances(parent_pareto_scores)
                child_agg = self.aggregate_instances(child_pareto_scores)
                accepted_iterations.append(
                    self._accept_child(
                        selected_module_id,
                        parent_prompt_configuration,
                        child_prompt_configuration,
                        child_pareto_scores,
                        parent_agg,
                        child_agg,
                    )
                )
                self._iteration_log.append({
                    "iteration": self._current_iteration,
                    "outcome": "accepted",
                    "reason": "Accepted by Pareto non-domination",
                    "before": parent_agg,
                    "after": child_agg,
                    "elapsed": time.perf_counter() - iter_start,
                })
            else:
                consecutive_rejections += 1
                self._iteration_log.append({
                    "iteration": self._current_iteration,
                    "outcome": "rejected",
                    "reason": f"Rejected (consecutive rejections: {consecutive_rejections}/{self.patience})",
                    "before": self.aggregate_instances(parent_pareto_scores),
                    "after": self.aggregate_instances(child_pareto_scores),
                    "elapsed": time.perf_counter() - iter_start,
                })

            if consecutive_rejections >= self.patience:
                self._iteration_log[-1]["reason"] = f"early stop (patience={self.patience})"
                return False

            return True

        self._run_loop_iteration(_one_iteration)
        if not self.pareto_score_table:
            raise DeepEvalError(
                "GEPA finished without any Pareto scores (empty score table). "
                "Common causes: empty feedback split, or the loop exited before "
                "the first scoring step ran."
            )
        best = self._best_by_aggregate()
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )
        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report

    async def a_execute(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, OptimizationReport]:
        """Asynchronous twin of execute_gepa()."""
        total_goldens = len(goldens)
        if total_goldens < 2:
            raise DeepEvalError(
                "GEPA prompt optimization requires at least 2 goldens, but "
                f"received {total_goldens}. Provide at least two goldens to "
                "run the optimizer."
            )

        if self.reflection_model is not None:
            self.scorer.optimizer_model = self.reflection_model
        if self.mutation_model is not None:
            self._rewriter.optimizer_model = self.mutation_model

        self._ensure_scorer()
        self.reset_state()

        d_feedback, d_pareto = split_goldens(
            goldens, self.pareto_size, random_state=self.random_state
        )

        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []
        consecutive_rejections = 0

        async def _one_iteration() -> bool:
            nonlocal accepted_iterations, consecutive_rejections

            if not d_feedback:
                return False

            iter_start = time.perf_counter()
            cur = self._current_iteration

            # Seed Pareto scores lazily on first iteration
            if not self.pareto_score_table:
                self._update_step(cur, f"Scoring seed prompt on {len(d_pareto)} pareto goldens...")
                self.pareto_score_table[root_prompt_configuration.id] = (
                    await self.scorer.a_score_pareto(
                        root_prompt_configuration, d_pareto
                    )
                )

            # 1. Pick prompt_configuration via Pareto
            parent_prompt_configuration = self._pick_prompt_configuration()

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            minibatch = self._draw_minibatch(d_feedback)

            # 4. Feedback
            self._update_step(cur, f"Gathering feedback on {len(minibatch)} goldens...")
            feedback_diagnosis = await self.scorer.a_get_minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            parent_minibatch_score = await self.scorer.a_score_minibatch(
                parent_prompt_configuration, minibatch
            )

            # 5. Rewrite
            self._update_step(cur, "Rewriting prompt from feedback...")
            child_prompt = await self._a_generate_child_prompt(
                selected_module_id, parent_prompt_configuration, feedback_diagnosis
            )

            if child_prompt is None:
                self._iteration_log.append({
                    "iteration": cur,
                    "outcome": "skipped",
                    "reason": "child == parent",
                    "before": None,
                    "after": None,
                    "elapsed": time.perf_counter() - iter_start,
                })
                return True

            # 6. Child prompt_configuration
            child_prompt_configuration = self._make_child(
                selected_module_id, parent_prompt_configuration, child_prompt
            )

            child_minibatch_score = await self.scorer.a_score_minibatch(
                child_prompt_configuration, minibatch
            )

            if child_minibatch_score <= parent_minibatch_score:
                parent_agg = self.aggregate_instances(
                    self.pareto_score_table[parent_prompt_configuration.id]
                )
                self._iteration_log.append({
                    "iteration": cur,
                    "outcome": "skipped",
                    "reason": f"Skipped (minibatch score did not improve)",
                    "before": parent_agg,
                    "after": child_minibatch_score,
                    "elapsed": time.perf_counter() - iter_start,
                })
                return True

            # 7. Evaluate child on the GLOBAL validation set (d_pareto)
            self._update_step(cur, f"Evaluating child on pareto set ({len(d_pareto)} goldens)...")
            child_pareto_scores = await self.scorer.a_score_pareto(
                child_prompt_configuration, d_pareto
            )
            parent_pareto_scores = self.pareto_score_table[
                parent_prompt_configuration.id
            ]

            # 8. Acceptance test (Pareto non-domination vs parent)
            accepted = self._should_accept_child(child_pareto_scores, parent_pareto_scores)

            if accepted:
                consecutive_rejections = 0
                parent_agg = self.aggregate_instances(parent_pareto_scores)
                child_agg = self.aggregate_instances(child_pareto_scores)
                accepted_iterations.append(
                    await self._a_accept_child(
                        selected_module_id,
                        parent_prompt_configuration,
                        child_prompt_configuration,
                        child_pareto_scores,
                        parent_agg,
                        child_agg,
                    )
                )
                self._iteration_log.append({
                    "iteration": cur,
                    "outcome": "accepted",
                    "reason": "Accepted by Pareto non-domination",
                    "before": parent_agg,
                    "after": child_agg,
                    "elapsed": time.perf_counter() - iter_start,
                })
            else:
                consecutive_rejections += 1
                self._iteration_log.append({
                    "iteration": cur,
                    "outcome": "rejected",
                    "reason": f"Rejected (consecutive rejections: {consecutive_rejections}/{self.patience})",
                    "before": self.aggregate_instances(parent_pareto_scores),
                    "after": self.aggregate_instances(child_pareto_scores),
                    "elapsed": time.perf_counter() - iter_start,
                })

            if consecutive_rejections >= self.patience:
                self._iteration_log[-1]["reason"] = f"early stop (patience={self.patience})"
                return False

            return True

        await self._a_run_loop_iteration(_one_iteration)
        if not self.pareto_score_table:
            raise DeepEvalError(
                "GEPA finished without any Pareto scores (empty score table). "
                "Common causes: empty feedback split, or the loop exited before "
                "the first scoring step ran."
            )
        best = self._best_by_aggregate()
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )
        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report

    ###################
    # State & helpers #
    ###################

    def reset_state(self) -> None:
        self.optimization_id = str(uuid.uuid4())
        self.prompt_configurations_by_id: Dict[
            PromptConfigurationId, PromptConfiguration
        ] = {}
        self.parents_by_id: Dict[
            PromptConfigurationId, Optional[PromptConfigurationId]
        ] = {}
        self.pareto_score_table: ScoreTable = {}
        # Accumulates one dict per iteration for the final summary table
        self._iteration_log: List[Dict] = []
        self._current_iteration: int = 0

    def _ensure_scorer(self) -> None:
        if self.scorer is None:
            raise DeepEvalError(
                "GEPARunner requires a `scorer`. "
                "Construct one (for example, Scorer) in "
                "PromptOptimizer and assign it to `runner.scorer`."
            )

    def _prompts_equivalent(
        self, old_prompt: Prompt, new_prompt: Prompt
    ) -> bool:
        """
        Compare two Prompts for GEPA acceptance purposes.

        This is used as:
            if self._prompts_equivalent(old, new):
                # reject child (treat as "no change")
                return None

        So:
        - Return True:  "do not accept this child"
        - Return False: "child is meaningfully different"

        Rules:
        - If the types must be the same for this check to be meaningful
        - For TEXT: compare text_template with whitespace trimmed
        - For LIST: compare messages_template (length, role, and content,
          with content whitespace trimmed).
        """

        # LIST prompts: compare messages
        if new_prompt.type == PromptType.LIST:
            old_msgs = old_prompt.messages_template
            new_msgs = new_prompt.messages_template
            if len(old_msgs) != len(new_msgs):
                return False

            for old_msg, new_msg in zip(old_msgs, new_msgs):
                if old_msg.role != new_msg.role:
                    return False
                if (old_msg.content or "").strip() != (
                    new_msg.content or ""
                ).strip():
                    return False

            return True

        # TEXT prompts: compare text_template
        old_txt = (old_prompt.text_template or "").strip()
        new_txt = (new_prompt.text_template or "").strip()
        return new_txt == old_txt

    def _add_prompt_configuration(
        self, prompt_configuration: PromptConfiguration
    ) -> None:
        self.prompt_configurations_by_id[prompt_configuration.id] = (
            prompt_configuration
        )
        self.parents_by_id[prompt_configuration.id] = (
            prompt_configuration.parent
        )

    def _best_by_aggregate(self) -> PromptConfiguration:
        totals = {
            prompt_configuration_id: self.aggregate_instances(vector)
            for prompt_configuration_id, vector in self.pareto_score_table.items()
        }

        chosen, tied, max_val = pick_best_with_ties(
            totals,
            self.parents_by_id,
            random_state=self.random_state,
            tie_tolerance=GEPA_TIE_TOLERANCE,
            policy=self.tie_breaker,
        )
        if self.status_callback is not None and len(tied) > 1:
            msg = (
                f"tie on aggregate={max_val:.4f} among {len(tied)} "
                f"prompt_configurations; using tie_breaker="
                f"{self.tie_breaker.value!r} selected {chosen}. "
                f"To change, set GEPA tie_breaker to one of: "
                f"{[t.value for t in self.TieBreaker]}."
            )
            self.status_callback(
                RunnerStatusType.TIE,
                detail=msg,
            )

        return self.prompt_configurations_by_id[chosen]

    def _pick_prompt_configuration(self) -> PromptConfiguration:
        selected_prompt_configuration_id = select_prompt_configuration_pareto(
            self.pareto_score_table, random_state=self.random_state
        )
        return self.prompt_configurations_by_id[
            selected_prompt_configuration_id
        ]

    def _draw_minibatch(
        self, d_feedback: Union[List["Golden"], List["ConversationalGolden"]]
    ) -> Union[List["Golden"], List["ConversationalGolden"]]:
        # Determine effective minibatch size, bounded by the
        # available feedback set.
        n_feedback = len(d_feedback)
        if n_feedback <= 0:
            return []

        size = min(self.minibatch_size, n_feedback)

        return [
            d_feedback[self.random_state.randrange(0, n_feedback)]
            for _ in range(size)
        ]

    async def _a_generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        feedback_diagnosis: ScorerDiagnosisResult,
    ) -> Optional[Prompt]:
        old_prompt = parent_prompt_configuration.prompts.get(
            selected_module_id, Prompt(text_template="")
        )

        new_prompt = await self._rewriter.a_rewrite(
            old_prompt=old_prompt,
            feedback_diagnosis=feedback_diagnosis,
        )

        if old_prompt.type != new_prompt.type or self._prompts_equivalent(
            old_prompt, new_prompt
        ):
            # don't accept if new prompt is the same as parent
            # or if the type somehow changed
            return None
        return new_prompt

    def _generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        feedback_diagnosis: ScorerDiagnosisResult,
    ) -> Optional[Prompt]:
        old_prompt = parent_prompt_configuration.prompts.get(
            selected_module_id, Prompt(text_template="")
        )

        new_prompt = self._rewriter.rewrite(
            old_prompt=old_prompt,
            feedback_diagnosis=feedback_diagnosis,
        )

        if old_prompt.type != new_prompt.type or self._prompts_equivalent(
            old_prompt, new_prompt
        ):
            # don't accept if new prompt is the same as parent
            # or if the type somehow changed
            return None
        return new_prompt

    def _make_child(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        child_prompt: Prompt,
    ) -> PromptConfiguration:
        child_prompt_configuration = PromptConfiguration.new(
            prompts=dict(parent_prompt_configuration.prompts),
            parent=parent_prompt_configuration.id,
        )
        child_prompt_configuration.prompts[selected_module_id] = child_prompt
        return child_prompt_configuration

    def _should_accept_child(
        self, child_scores: List[float], parent_scores: List[float]
    ) -> bool:
        if _is_dominated(candidate_scores=child_scores, other_scores=parent_scores, min_delta=GEPA_MIN_DELTA):
            return False

        current_archive_scores = list(self.pareto_score_table.values())

        for existing_scores in current_archive_scores:
            if _is_dominated(candidate_scores=child_scores, other_scores=existing_scores, min_delta=GEPA_MIN_DELTA):
                return False

        return True

    def _accept_child(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        child_prompt_configuration: PromptConfiguration,
        child_pareto_scores: List[float],
        parent_agg_score: float,
        child_agg_score: float,
    ) -> AcceptedIterationDict:
        self._add_prompt_configuration(child_prompt_configuration)
        self.pareto_score_table[child_prompt_configuration.id] = child_pareto_scores

        ids_to_remove = []
        for config_id, scores in self.pareto_score_table.items():
            if config_id == child_prompt_configuration.id:
                continue
            if _is_dominated(candidate_scores=scores, other_scores=child_pareto_scores, min_delta=GEPA_MIN_DELTA):
                ids_to_remove.append(config_id)

        for rid in ids_to_remove:
            del self.pareto_score_table[rid]

        return AcceptedIterationDict(
            parent=parent_prompt_configuration.id,
            child=child_prompt_configuration.id,
            module=selected_module_id,
            before=parent_agg_score,
            after=child_agg_score,
        )

    async def _a_accept_child(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        child_prompt_configuration: PromptConfiguration,
        child_pareto_scores: List[float],
        parent_agg_score: float,
        child_agg_score: float,
    ) -> AcceptedIterationDict:
        self._add_prompt_configuration(child_prompt_configuration)
        self.pareto_score_table[child_prompt_configuration.id] = child_pareto_scores

        ids_to_remove = []
        for config_id, scores in self.pareto_score_table.items():
            if config_id == child_prompt_configuration.id:
                continue
            if _is_dominated(candidate_scores=scores, other_scores=child_pareto_scores, min_delta=GEPA_MIN_DELTA):
                ids_to_remove.append(config_id)

        for rid in ids_to_remove:
            del self.pareto_score_table[rid]

        return AcceptedIterationDict(
            parent=parent_prompt_configuration.id,
            child=child_prompt_configuration.id,
            module=selected_module_id,
            before=parent_agg_score,
            after=child_agg_score,
        )


    def _update_step(self, iteration: int, label: str) -> None:
        """Update the sub-step row on the outer progress bar."""
        if self.step_callback is not None:
            self.step_callback(label)

    def _update_progress(
        self,
        total_iterations: int,
        iteration: int,
        remaining_iterations: int,
    ):
        if self.status_callback is not None:
            detail = (
                f"(iterations={total_iterations}) "
                f"• iteration {iteration}/{total_iterations} "
                f"• remaining={remaining_iterations}"
            )
            self.status_callback(
                RunnerStatusType.PROGRESS,
                step_index=iteration,
                total_steps=total_iterations,
                detail=detail,
            )

    def _update_error(
        self, total_iterations: int, iteration: int, exc: Exception
    ):
        # Report a user facing error event
        if self.status_callback is not None:
            detail = (
                f"(iterations={total_iterations}) "
                f"• error {exc.__class__.__name__}: {exc} "
                f"• halted at iteration {iteration}"
            )
            self.status_callback(
                RunnerStatusType.ERROR,
                step_index=iteration,
                total_steps=total_iterations,
                detail=detail,
            )

    def _run_loop_iteration(
        self,
        gepa_iteration: Callable[[], bool],
    ) -> None:
        total_iterations = self.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(total_iterations, iteration, remaining_iterations)
        while remaining_iterations > 0:
            iteration += 1
            self._current_iteration = iteration
            try:
                ok = gepa_iteration()
            except Exception as exc:
                self._update_error(total_iterations, iteration, exc)
                raise
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations
            )

    async def _a_run_loop_iteration(
        self,
        a_gepa_iteration: Callable[[], Awaitable[bool]],
    ) -> None:
        total_iterations = self.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(total_iterations, iteration, remaining_iterations)
        while remaining_iterations > 0:
            iteration += 1
            self._current_iteration = iteration
            try:
                ok = await a_gepa_iteration()
            except Exception as exc:
                self._update_error(total_iterations, iteration, exc)
                raise
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations
            )

    def generate_summary_table(self, report: OptimizationReport) -> List[Table]:
        """Generates GEPA-specific evolutionary iteration and Pareto tables."""
        _PURPLE = "rgb(106,0,255)"
        _GREEN  = "rgb(25,227,160)"
        _RED    = "rgb(255,85,85)"
        _DIM    = "rgb(55,65,81)"

        tables = []
        iteration_log = getattr(self, "_iteration_log", [])

        # 1. Iteration Table
        iter_table = Table(
            title=f"✨ [{_PURPLE}]{self.name}[/] Evolutionary Mutations",
            box=box.ROUNDED, border_style=_PURPLE, header_style=f"bold {_PURPLE}", show_lines=True, expand=True
        )
        iter_table.add_column("#", style="bold white", justify="right", no_wrap=True)
        iter_table.add_column("Outcome", justify="center", no_wrap=True)
        iter_table.add_column("Before", justify="right", no_wrap=True)
        iter_table.add_column("After", justify="right", no_wrap=True)
        iter_table.add_column("Δ Score", justify="right", no_wrap=True)
        iter_table.add_column("Note", style=f"{_DIM}", no_wrap=False)
        iter_table.add_column("Time", justify="right", no_wrap=True)

        for entry in iteration_log:
            i = str(entry["iteration"])
            outcome = entry["outcome"]
            before = entry.get("before")
            after = entry.get("after")
            reason = entry.get("reason", "")
            elapsed = entry.get("elapsed", 0.0)

            if outcome == "accepted":
                outcome_cell = f"[{_GREEN}]✔ accepted[/]"
            elif outcome == "rejected":
                outcome_cell = f"[{_RED}]✘ rejected[/]"
            else:
                outcome_cell = f"[{_DIM}]↷ skipped[/]"

            before_cell = f"{before:.4f}" if before is not None else "—"
            after_cell  = f"{after:.4f}" if after is not None else "—"

            if before is not None and after is not None:
                delta = after - before
                sign = "+" if delta >= 0 else ""
                color = _GREEN if delta > 0 else (_RED if delta < 0 else _DIM)
                delta_cell = f"[{color}]{sign}{delta:.4f}[/]"
            else:
                delta_cell = "—"

            time_cell = f"[{_DIM}]{elapsed:.2f}s[/]"
            iter_table.add_row(i, outcome_cell, before_cell, after_cell, delta_cell, reason, time_cell)
        
        tables.append(iter_table)

        # 2. Pareto Table
        if report and report.pareto_scores:
            pareto_table = Table(
                title=f"[{_PURPLE}]Final Pareto Archive[/]",
                box=box.HORIZONTALS, border_style=_PURPLE, header_style=f"bold {_PURPLE}", show_lines=True, expand=True
            )
            pareto_table.add_column("Config ID", style="white", no_wrap=True)
            pareto_table.add_column("Role", justify="center", no_wrap=True)
            pareto_table.add_column("Scores", no_wrap=False)
            pareto_table.add_column("Aggregate", justify="right", no_wrap=True)

            best_id = report.best_id
            for cid, scores in report.pareto_scores.items():
                is_root = report.parents.get(cid) is None
                role = f"[{_PURPLE}]root[/]" if is_root else f"[{_DIM}]child[/]"
                is_best = cid == best_id

                short_id = cid[:8] + "…"
                if is_best:
                    short_id = f"[bold {_GREEN}]{short_id} ★[/]"

                if len(scores) > 6:
                    score_strs = [f"{s:.3f}" for s in scores[:3]] + ["..."] + [f"{s:.3f}" for s in scores[-3:]]
                else:
                    score_strs = [f"{s:.3f}" for s in scores]
                scores_cell = f"[{_DIM}][{', '.join(score_strs)}][/]"

                agg = sum(scores) / len(scores) if scores else 0.0
                agg_color = _GREEN if is_best else "white"
                agg_cell  = f"[{agg_color}]{agg:.4f}[/]"

                pareto_table.add_row(short_id, role, scores_cell, agg_cell)
            
            tables.append(pareto_table)

        return tables