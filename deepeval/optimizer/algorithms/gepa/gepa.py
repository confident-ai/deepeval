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

from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval.errors import DeepEvalError
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
    frequency_weights,
    pareto_frontier,
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
        tie_breaker: TieBreaker = TieBreaker.PREFER_CHILD,
        aggregate_instances: Aggregator = mean_of_all,
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

        # Optimizer model used by the rewriter for prompt mutation.
        # Set by PromptOptimizer.
        self.optimizer_model: Optional["DeepEvalBaseLLM"] = None

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

        def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not d_feedback:
                return False

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
            feedback_text = self.scorer.get_minibatch_feedback(
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
            parent_score = self.scorer.score_minibatch(
                parent_prompt_configuration, minibatch
            )
            child_score = self.scorer.score_minibatch(
                child_prompt_configuration, minibatch
            )

            # 8. Acceptance test
            accepted = self._should_accept_child(parent_score, child_score)
            if accepted:
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

        async def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not d_feedback:
                return False

            iter_start = time.perf_counter()

            # Seed Pareto scores lazily on first iteration
            if not self.pareto_score_table:
                t0 = time.perf_counter()
                self.pareto_score_table[root_prompt_configuration.id] = (
                    await self.scorer.a_score_pareto(
                        root_prompt_configuration, d_pareto
                    )
                )
                print(
                    f"[DEBUG] Initial pareto scoring ({len(d_pareto)} goldens): {time.perf_counter() - t0:.2f}s"
                )

            # 1. Pick prompt_configuration via Pareto
            parent_prompt_configuration = self._pick_prompt_configuration()

            # 2. Single module id
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            # 3. Draw minibatch
            minibatch = self._draw_minibatch(d_feedback)
            print(f"[DEBUG] Minibatch size: {len(minibatch)}")

            # 4. Feedback
            t0 = time.perf_counter()
            feedback_text = await self.scorer.a_get_minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )
            print(f"[DEBUG] Get feedback: {time.perf_counter() - t0:.2f}s")

            # 5. Rewrite
            t0 = time.perf_counter()
            child_prompt = await self._a_generate_child_prompt(
                selected_module_id, parent_prompt_configuration, feedback_text
            )
            print(f"[DEBUG] Rewrite prompt: {time.perf_counter() - t0:.2f}s")
            if child_prompt is None:
                print(f"[DEBUG] Child prompt same as parent, skipping")
                return True

            # 6. Child prompt_configuration
            child_prompt_configuration = self._make_child(
                selected_module_id, parent_prompt_configuration, child_prompt
            )

            # 7. Evaluate parent/child on minibatch
            t0 = time.perf_counter()
            parent_score = await self.scorer.a_score_minibatch(
                parent_prompt_configuration, minibatch
            )
            print(
                f"[DEBUG] Score parent on minibatch: {time.perf_counter() - t0:.2f}s (score={parent_score:.4f})"
            )

            t0 = time.perf_counter()
            child_score = await self.scorer.a_score_minibatch(
                child_prompt_configuration, minibatch
            )
            print(
                f"[DEBUG] Score child on minibatch: {time.perf_counter() - t0:.2f}s (score={child_score:.4f})"
            )

            # 8. Acceptance test
            accepted = self._should_accept_child(parent_score, child_score)
            print(
                f"[DEBUG] Acceptance: {'ACCEPTED' if accepted else 'REJECTED'}"
            )
            if accepted:
                t0 = time.perf_counter()
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
                print(
                    f"[DEBUG] Accept child (pareto scoring): {time.perf_counter() - t0:.2f}s"
                )

            print(
                f"[DEBUG] Total iteration time: {time.perf_counter() - iter_start:.2f}s\n"
            )
            return True

        await self._a_run_loop_iteration(_one_iteration)
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
        # Log Pareto selection details
        all_candidates = list(self.pareto_score_table.keys())
        print(f"[DEBUG] Pareto Selection:")
        print(f"  - Total candidates in pool: {len(all_candidates)}")

        # Show score table
        print(f"  - Score table (per-instance scores):")
        for cid, scores in self.pareto_score_table.items():
            is_root = self.parents_by_id.get(cid) is None
            label = (
                "(root)"
                if is_root
                else f"(child of {self.parents_by_id.get(cid)[:8]}...)"
            )
            mean_score = sum(scores) / len(scores) if scores else 0
            print(
                f"      {cid[:8]}... {label}: {[round(s, 3) for s in scores]} (mean={mean_score:.3f})"
            )

        # Show Pareto frontier
        frontier = pareto_frontier(all_candidates, self.pareto_score_table)
        print(f"  - Pareto frontier ({len(frontier)} non-dominated):")
        for cid in frontier:
            print(f"      {cid[:8]}...")

        # Show frequency weights
        freq = frequency_weights(self.pareto_score_table)
        print(f"  - Frequency weights (how often each wins an instance):")
        for cid, weight in freq.items():
            print(f"      {cid[:8]}...: {weight}")

        # Do the selection
        selected_prompt_configuration_id = select_prompt_configuration_pareto(
            self.pareto_score_table, random_state=self.random_state
        )
        print(f"  - Selected: {selected_prompt_configuration_id[:8]}...\n")

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
        feedback_text: str,
    ) -> Optional[Prompt]:
        old_prompt = parent_prompt_configuration.prompts.get(
            selected_module_id, Prompt(text_template="")
        )

        new_prompt = await self._rewriter.a_rewrite(
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
        return child_score >= parent_score + max(GEPA_MIN_DELTA, jitter)

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
            self.scorer.score_pareto(child_prompt_configuration, d_pareto)
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
            await self.scorer.a_score_pareto(
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
            try:
                ok = gepa_iteration()
            except Exception as exc:
                # Report a user facing error event and halt optimization.
                self._update_error(total_iterations, iteration, exc)
                break
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
            try:
                ok = await a_gepa_iteration()
            except Exception as exc:
                # Report a user facing error event and halt optimization.
                self._update_error(total_iterations, iteration, exc)
                break
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations
            )
