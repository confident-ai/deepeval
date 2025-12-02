from __future__ import annotations
import uuid
import random
import time

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

from deepeval.errors import DeepEvalError
from deepeval.optimization.aggregates import Aggregator, mean_of_all
from deepeval.optimization.types import (
    AcceptedIterationDict,
    PromptConfiguration,
    PromptConfigurationId,
    ModuleId,
    ScoreTable,
    ScoringAdapter,
    OptimizationResult,
    RunnerStatusType,
    RunnerStatusCallbackProtocol,
)
from deepeval.optimization.utils import (
    split_goldens,
    build_prompt_config_snapshots,
)
from deepeval.optimization.policies import (
    pick_best_with_ties,
    select_prompt_configuration_pareto,
)
from deepeval.prompt.api import PromptType
from deepeval.prompt.prompt import Prompt
from deepeval.optimization.mutations.prompt_rewriter import (
    PromptRewriter,
)
from .configs import GEPAConfig


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class GEPARunner:
    """
    GEPA loop with sync/async execution.

    This runner is intentionally low level and does not know about metrics,
    models, or async configs. It relies on a preconfigured
    ScoringAdapter and PromptRewriter, which are typically constructed by
    the higher-level PromptOptimizer.
    """

    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        *,
        config: GEPAConfig,
        aggregate_instances: Aggregator = mean_of_all,
        scoring_adapter: Optional[ScoringAdapter] = None,
    ) -> None:
        self.config = config
        self.aggregate_instances = aggregate_instances
        self.scoring_adapter = scoring_adapter

        # random seeded from config is used for splits, sampling, and tie-breaking.
        self.random_state = random.Random(config.random_seed)

        # runtime state to be reset between runs
        self.reset_state()

        # Status callback set by PromptOptimizer:
        #   (kind, step_index, total_steps, detail) -> None
        self.status_callback: Optional[RunnerStatusCallbackProtocol] = None

        # Model callback used by the rewriter set by PromptOptimizer.
        self.model_callback: Optional[
            Callable[
                ...,
                Union[
                    str,
                    Dict,
                    Tuple[Union[str, Dict], float],
                ],
            ]
        ] = None

        # lazy loaded
        self._rewriter: Optional[PromptRewriter] = None

    ##############
    # Public API #
    ##############

    def execute(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        """Synchronous GEPA run from a full list of goldens (splits internally)."""
        total_goldens = len(goldens)
        if total_goldens < 2:
            raise DeepEvalError(
                "GEPA prompt optimization requires at least 2 goldens, but "
                f"received {total_goldens}. Provide at least two goldens to "
                "run the optimizer."
            )

        self._ensure_scoring_adapter()
        self._ensure_rewriter()
        self.reset_state()

        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, random_state=self.random_state
        )

        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []

        def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not d_feedback:
                return False

            # Seed Pareto scores lazily on first iteration
            if not self.pareto_score_table:
                self.pareto_score_table[root_prompt_configuration.id] = (
                    self.scoring_adapter.score_on_pareto(
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
            feedback_text = self.scoring_adapter.minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            # 5. Rewrite
            child_prompt = self._generate_child_prompt(
                selected_module_id, parent_prompt_configuration, feedback_text
            )
            if child_prompt is None:
                # Child prompt matched parent; skip this iteration.
                return True

            # 6. Child prompt_configuration
            child_prompt_configuration = self._make_child(
                selected_module_id, parent_prompt_configuration, child_prompt
            )

            # 7. Evaluate parent/child on minibatch
            parent_score = self.scoring_adapter.minibatch_score(
                parent_prompt_configuration, minibatch
            )
            child_score = self.scoring_adapter.minibatch_score(
                child_prompt_configuration, minibatch
            )

            # 8. Acceptance test
            if self._should_accept_child(parent_score, child_score):
                accepted_iterations.append(
                    self._accept_child(
                        selected_module_id,
                        parent_prompt_configuration,
                        child_prompt_configuration,
                        d_pareto,
                        parent_score,
                        child_score,
                    )
                )

            return True

        self._run_loop_iteration(_one_iteration)
        best = self._best_by_aggregate()
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report.as_dict()

    async def a_execute(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        """Asynchronous twin of execute_gepa()."""
        total_goldens = len(goldens)
        if total_goldens < 2:
            raise DeepEvalError(
                "GEPA prompt optimization requires at least 2 goldens, but "
                f"received {total_goldens}. Provide at least two goldens to "
                "run the optimizer."
            )

        self._ensure_scoring_adapter()
        self._ensure_rewriter()
        self.reset_state()

        d_feedback, d_pareto = split_goldens(
            goldens, self.config.pareto_size, random_state=self.random_state
        )

        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []

        async def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not d_feedback:
                return False

            # Seed Pareto scores lazily on first iteration
            if not self.pareto_score_table:
                self.pareto_score_table[root_prompt_configuration.id] = (
                    await self.scoring_adapter.a_score_on_pareto(
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
            feedback_text = await self.scoring_adapter.a_minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            # 5. Rewrite
            child_prompt = await self._a_generate_child_prompt(
                selected_module_id, parent_prompt_configuration, feedback_text
            )
            if child_prompt is None:
                # Child prompt matched parent; skip this iteration.
                return True

            # 6. Child prompt_configuration
            child_prompt_configuration = self._make_child(
                selected_module_id, parent_prompt_configuration, child_prompt
            )

            # 7. Evaluate parent/child on minibatch
            parent_score = await self.scoring_adapter.a_minibatch_score(
                parent_prompt_configuration, minibatch
            )
            child_score = await self.scoring_adapter.a_minibatch_score(
                child_prompt_configuration, minibatch
            )

            # 8. Acceptance test
            if self._should_accept_child(parent_score, child_score):
                accepted_iterations.append(
                    await self._a_accept_child(
                        selected_module_id,
                        parent_prompt_configuration,
                        child_prompt_configuration,
                        d_pareto,
                        parent_score,
                        child_score,
                    )
                )
            return True

        await self._a_run_loop_iteration(_one_iteration)
        best = self._best_by_aggregate()
        prompt_config_snapshots = build_prompt_config_snapshots(
            self.prompt_configurations_by_id
        )
        report = OptimizationResult(
            optimization_id=self.optimization_id,
            best_id=best.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=prompt_config_snapshots,
        )
        return best.prompts[self.SINGLE_MODULE_ID], report.as_dict()

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

    def _ensure_scoring_adapter(self) -> None:
        if self.scoring_adapter is None:
            raise DeepEvalError(
                "GEPARunner requires a `scoring_adapter`. "
                "Construct one (for example, DeepEvalScoringAdapter) in "
                "PromptOptimizer and assign it to `runner.scoring_adapter`."
            )

    def _ensure_rewriter(self) -> None:
        if self._rewriter is not None:
            return

        # For now, always use the basic PromptRewriter. Additional
        # variants (e.g. for GEPA Alg. 4 crossover) can be introduced
        # later
        self._rewriter = PromptRewriter()

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
            tie_tolerance=float(self.config.tie_tolerance),
            policy=self.config.tie_breaker,
        )
        if self.status_callback is not None and len(tied) > 1:
            msg = (
                f"tie on aggregate={max_val:.4f} among {len(tied)} "
                f"prompt_configurations; using tie_breaker="
                f"{self.config.tie_breaker.value!r} selected {chosen}. "
                f"To change, set GEPAConfig.tie_breaker to one of: "
                f"{[t.value for t in self.config.TieBreaker]} "
                f"(tie_tolerance={float(self.config.tie_tolerance):g})."
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
        # Determine effective minibatch size from GEPAConfig, bounded by the
        # available feedback set.
        n_feedback = len(d_feedback)
        if n_feedback <= 0:
            return []

        if self.config.minibatch_size is not None:
            size = self.config.minibatch_size
        else:
            # Dynamic sizing from ratio, bounded between min and max.
            dynamic = max(
                1, int(round(n_feedback * self.config.minibatch_ratio))
            )
            size = max(
                self.config.minibatch_min_size,
                min(dynamic, self.config.minibatch_max_size),
            )

        size = max(1, min(size, n_feedback))

        return [
            d_feedback[self.random_state.randrange(0, n_feedback)]
            for _ in range(size)
        ]

    async def _a_generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        feedback_text: str,
    ) -> Optional[Prompt]:
        old_prompt = parent_prompt_configuration.prompts.get(
            selected_module_id, Prompt(text_template="")
        )

        new_prompt = await self._rewriter.a_rewrite(
            model_callback=self.model_callback,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
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
        feedback_text: str,
    ) -> Optional[Prompt]:
        old_prompt = parent_prompt_configuration.prompts.get(
            selected_module_id, Prompt(text_template="")
        )

        new_prompt = self._rewriter.rewrite(
            model_callback=self.model_callback,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
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
        self, parent_score: float, child_score: float
    ) -> bool:
        jitter = 1e-6
        return child_score >= parent_score + max(self.config.min_delta, jitter)

    def _accept_child(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        child_prompt_configuration: PromptConfiguration,
        d_pareto: Union[List["Golden"], List["ConversationalGolden"]],
        parent_score: float,
        child_score: float,
    ) -> AcceptedIterationDict:
        self._add_prompt_configuration(child_prompt_configuration)
        self.pareto_score_table[child_prompt_configuration.id] = (
            self.scoring_adapter.score_on_pareto(
                child_prompt_configuration, d_pareto
            )
        )

        return AcceptedIterationDict(
            parent=parent_prompt_configuration.id,
            child=child_prompt_configuration.id,
            module=selected_module_id,
            before=parent_score,
            after=child_score,
        )

    async def _a_accept_child(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        child_prompt_configuration: PromptConfiguration,
        d_pareto: Union[List["Golden"], List["ConversationalGolden"]],
        parent_score: float,
        child_score: float,
    ) -> AcceptedIterationDict:
        self._add_prompt_configuration(child_prompt_configuration)
        self.pareto_score_table[child_prompt_configuration.id] = (
            await self.scoring_adapter.a_score_on_pareto(
                child_prompt_configuration, d_pareto
            )
        )

        return AcceptedIterationDict(
            parent=parent_prompt_configuration.id,
            child=child_prompt_configuration.id,
            module=selected_module_id,
            before=parent_score,
            after=child_score,
        )

    def _update_progress(
        self,
        total_iterations: int,
        iteration: int,
        remaining_iterations: int,
        elapsed: float,
    ):
        if self.status_callback is not None:
            detail = (
                f"(iterations={total_iterations}) "
                f"• iteration {iteration}/{total_iterations} "
                f"• {elapsed:.2f}s • remaining={remaining_iterations}"
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
        total_iterations = self.config.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(
            total_iterations, iteration, remaining_iterations, 0
        )
        while remaining_iterations > 0:
            iteration += 1
            start_time = time.perf_counter()
            try:
                ok = gepa_iteration()
            except Exception as exc:
                # Report a user facing error event and halt optimization.
                self._update_error(total_iterations, iteration, exc)
                break
            elapsed = time.perf_counter() - start_time
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations, elapsed
            )

    async def _a_run_loop_iteration(
        self,
        a_gepa_iteration: Callable[[], Awaitable[bool]],
    ) -> None:
        total_iterations = self.config.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(
            total_iterations, iteration, remaining_iterations, 0
        )
        while remaining_iterations > 0:
            iteration += 1
            start_time = time.perf_counter()
            try:
                ok = await a_gepa_iteration()
            except Exception as exc:
                # Report a user facing error event and halt optimization.
                self._update_error(total_iterations, iteration, exc)
                break
            elapsed = time.perf_counter() - start_time
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations, elapsed
            )
