# - COPRO cooperative 0-shot variant:
#   - Works on a single set of goldens (no D_pareto split).
#   - Maintains a bounded population of candidate prompts
#     (size controlled by `population_size`).
#   - At each iteration:
#       - Select a parent via epsilon-greedy on mean minibatch score.
#       - Sample a minibatch of goldens for scoring.
#       - Compute feedback once for the parent + minibatch.
#       - Propose multiple child prompts cooperatively from the same parent
#         (up to `proposals_per_step` children).
#       - For each child, accept it if its minibatch score improves on the
#         parent by at least `min_delta`, add it to the pool, and prune
#         low-scoring candidates if the population exceeds `population_size`.
#   - Uses `full_eval_every` (if set) to periodically re-score the current
#     best candidate on the full golden set.

from __future__ import annotations

import random
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from deepeval.errors import DeepEvalError
from deepeval.optimization.aggregates import Aggregator, mean_of_all
from deepeval.optimization.types import (
    AcceptedIterationDict,
    ModuleId,
    OptimizationResult,
    PromptConfiguration,
    PromptConfigurationId,
    RunnerStatusCallbackProtocol,
    RunnerStatusType,
    ScoreTable,
    ScoringAdapter,
)
from deepeval.optimization.utils import (
    build_prompt_config_snapshots,
)
from deepeval.prompt.api import PromptType
from deepeval.prompt.prompt import Prompt
from deepeval.optimization.mutations.prompt_rewriter import PromptRewriter

from .configs import COPROConfig

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from deepeval.dataset.golden import ConversationalGolden, Golden


class COPRORunner:
    """
    COPRO style cooperative prompt optimization loop with sync/async execution.

    This runner is intentionally low level and does not know about metrics,
    models, or async configs. It relies on a preconfigured ScoringAdapter and
    PromptRewriter, which are typically constructed by PromptOptimizer.

    - Optimizes a single Prompt (instruction) against a list of Goldens.
    - Uses mini-batches of goldens for trial scoring and epsilon-greedy
      selection over prompt candidates based on mean minibatch scores,
      extended with cooperative proposals:
        - At each iteration, a parent candidate is selected.
        - A shared feedback string is computed on a minibatch.
        - Multiple child prompts are proposed from that parent using the
          same feedback but different LLM samples.
        - Any child whose minibatch score improves over the parent by at
          least ``min_delta`` is added to the candidate pool.
    """

    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        *,
        config: COPROConfig,
        aggregate_instances: Aggregator = mean_of_all,
        scoring_adapter: Optional[ScoringAdapter] = None,
    ) -> None:
        self.config = config
        self.aggregate_instances = aggregate_instances
        self.scoring_adapter = scoring_adapter

        # Random seeded from config is used for minibatch sampling and
        # epsilon-greedy candidate selection.
        self.random_state = random.Random(config.random_seed)

        self.random_state = random.Random(config.random_seed)

        # Runtime state to be reset between runs
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

        # Lazy-loaded PromptRewriter set by PromptOptimizer
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
        """
        Synchronous COPRO run from a full list of goldens.

        The full goldens set is used both for mini-batched scoring during
        optimization and for a final full evaluation of the best candidate.
        """
        total_goldens = len(goldens)
        if total_goldens < 1:
            raise DeepEvalError(
                "COPRO prompt optimization requires at least 1 golden, but "
                f"received {total_goldens}. Provide at least one golden to run "
                "the optimizer."
            )

        self._ensure_scoring_adapter()
        self._ensure_rewriter()
        self.reset_state()

        # Seed candidate pool with the root prompt configuration.
        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        # Add root candidate to the pool, but defer its first minibatch
        # evaluation until the first iteration so that any long running
        # model calls happen under the main loop (with progress updates).
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []
        self.trial_index = 0

        def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not goldens:
                return False

            # Lazily seed with a minibatch score for the root
            # candidate on the first iteration.
            if not self._minibatch_score_counts:
                seed_minibatch = self._draw_minibatch(goldens)
                root_score = self.scoring_adapter.minibatch_score(
                    root_prompt_configuration, seed_minibatch
                )
                self._record_minibatch_score(
                    root_prompt_configuration.id, root_score
                )

            # 1. Choose which candidate prompt to mutate.
            parent_prompt_configuration = self._select_candidate()
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            minibatch = self._draw_minibatch(goldens)

            # Compute shared feedback for this parent/minibatch that will be
            # used by all cooperative child proposals.
            feedback_text = self.scoring_adapter.minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            before_mean = self._mean_minibatch_score(
                parent_prompt_configuration.id
            )
            jitter = 1e-6
            min_delta = max(self.config.min_delta, jitter)

            # 2. Generate multiple cooperative child prompts and evaluate them.
            num_proposals = int(self.config.proposals_per_step)
            for _ in range(num_proposals):
                child_prompt = self._generate_child_prompt(
                    selected_module_id,
                    parent_prompt_configuration,
                    feedback_text,
                )
                if child_prompt is None:
                    # No child, nothing more to do this iteration
                    continue

                child_prompt_configuration = self._make_child(
                    selected_module_id,
                    parent_prompt_configuration,
                    child_prompt,
                )

                child_score = self.scoring_adapter.minibatch_score(
                    child_prompt_configuration, minibatch
                )

                # 3. Evaluate & decide whether to accept the child.
                if child_score >= before_mean + min_delta:
                    # Accept: add to pool, update surrogate stats, and record iteration.
                    self._add_prompt_configuration(child_prompt_configuration)
                    self._record_minibatch_score(
                        child_prompt_configuration.id, child_score
                    )

                    accepted_iterations.append(
                        AcceptedIterationDict(
                            parent=parent_prompt_configuration.id,
                            child=child_prompt_configuration.id,
                            module=selected_module_id,
                            before=before_mean,
                            after=child_score,
                        )
                    )
                # else: reject; do not add child to the candidate pool.

            self.trial_index += 1
            if (
                self.config.full_eval_every is not None
                and self.trial_index % self.config.full_eval_every == 0
            ):
                self._full_evaluate_best(goldens)

            return True

        self._run_loop_iteration(_one_iteration)

        # Ensure at least one candidate has been fully evaluated.
        if not self.pareto_score_table:
            self._full_evaluate_best(goldens)

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
        """
        Asynchronous twin of execute().
        """
        total_goldens = len(goldens)
        if total_goldens < 1:
            raise DeepEvalError(
                "COPRO prompt optimization requires at least 1 golden, but "
                f"received {total_goldens}. Provide at least one golden to run "
                "the optimizer."
            )

        self._ensure_scoring_adapter()
        self._ensure_rewriter()
        self.reset_state()

        seed_prompts_by_module = {self.SINGLE_MODULE_ID: prompt}
        root_prompt_configuration = PromptConfiguration.new(
            prompts=dict(seed_prompts_by_module)
        )
        # Add root candidate to the pool, but defer its first minibatch
        # evaluation until the first iteration so that any long running
        # model calls happen under the main loop (with progress updates).
        self._add_prompt_configuration(root_prompt_configuration)

        accepted_iterations: List[Dict] = []
        self.trial_index = 0

        async def _one_iteration() -> bool:
            nonlocal accepted_iterations

            if not goldens:
                return False

            # Lazily seed with a minibatch score for the root
            # candidate on the first iteration.
            if not self._minibatch_score_counts:
                seed_minibatch = self._draw_minibatch(goldens)
                root_score = await self.scoring_adapter.a_minibatch_score(
                    root_prompt_configuration, seed_minibatch
                )
                self._record_minibatch_score(
                    root_prompt_configuration.id, root_score
                )

            parent_prompt_configuration = self._select_candidate()
            selected_module_id: ModuleId = self.SINGLE_MODULE_ID

            minibatch = self._draw_minibatch(goldens)

            feedback_text = await self.scoring_adapter.a_minibatch_feedback(
                parent_prompt_configuration, selected_module_id, minibatch
            )

            before_mean = self._mean_minibatch_score(
                parent_prompt_configuration.id
            )
            jitter = 1e-6
            min_delta = max(self.config.min_delta, jitter)

            num_proposals = int(self.config.proposals_per_step)
            for _ in range(num_proposals):
                child_prompt = await self._a_generate_child_prompt(
                    selected_module_id,
                    parent_prompt_configuration,
                    feedback_text,
                )
                if child_prompt is None:
                    continue

                child_prompt_configuration = self._make_child(
                    selected_module_id,
                    parent_prompt_configuration,
                    child_prompt,
                )

                child_score = await self.scoring_adapter.a_minibatch_score(
                    child_prompt_configuration, minibatch
                )

                if child_score >= before_mean + min_delta:
                    self._add_prompt_configuration(child_prompt_configuration)
                    self._record_minibatch_score(
                        child_prompt_configuration.id, child_score
                    )

                    accepted_iterations.append(
                        AcceptedIterationDict(
                            parent=parent_prompt_configuration.id,
                            child=child_prompt_configuration.id,
                            module=selected_module_id,
                            before=before_mean,
                            after=child_score,
                        )
                    )

            self.trial_index += 1
            if (
                self.config.full_eval_every is not None
                and self.trial_index % self.config.full_eval_every == 0
            ):
                await self._a_full_evaluate_best(goldens)

            return True

        await self._a_run_loop_iteration(_one_iteration)

        if not self.pareto_score_table:
            await self._a_full_evaluate_best(goldens)

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
        # For COPRO we reuse the same field name as GEPA for full evaluation scores.
        self.pareto_score_table: ScoreTable = {}

        # Surrogate stats: running mean minibatch scores per candidate.
        self._minibatch_score_sums: Dict[PromptConfigurationId, float] = {}
        self._minibatch_score_counts: Dict[PromptConfigurationId, int] = {}

        # Trial counter (used for full_eval_every).
        self.trial_index: int = 0

    def _ensure_scoring_adapter(self) -> None:
        if self.scoring_adapter is None:
            raise DeepEvalError(
                "COPRORunner requires a `scoring_adapter`. "
                "Construct one (for example, DeepEvalScoringAdapter) in "
                "PromptOptimizer and assign it to `runner.scoring_adapter`."
            )

    def _ensure_rewriter(self) -> None:
        if self._rewriter is not None:
            return

        # Default basic PromptRewriter; PromptOptimizer can override this and
        # pass a configured instance (e.g. with list-mutation config).
        self._rewriter = PromptRewriter(
            max_chars=self.config.rewrite_instruction_max_chars,
            random_state=self.random_state,
        )

    def _prompts_equivalent(
        self,
        old_prompt: Prompt,
        new_prompt: Prompt,
    ) -> bool:
        """
        Compare two Prompts for optimization purposes.

        We treat a child as "no change" if:
        - The types differ, or
        - For TEXT: trimmed text_template matches.
        - For LIST: messages_template length, roles, and trimmed content match.
        """

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

        old_txt = (old_prompt.text_template or "").strip()
        new_txt = (new_prompt.text_template or "").strip()
        return new_txt == old_txt

    def _add_prompt_configuration(
        self,
        prompt_configuration: PromptConfiguration,
    ) -> None:
        """
        Add a candidate to the active pool and, if a population limit is set,
        prune the worst-scoring candidates to enforce it.
        """
        self.prompt_configurations_by_id[prompt_configuration.id] = (
            prompt_configuration
        )
        self.parents_by_id[prompt_configuration.id] = (
            prompt_configuration.parent
        )

        # If we exceed the population size, iteratively prune the worst
        # (by mean minibatch score), never removing the current best.
        while (
            len(self.prompt_configurations_by_id) > self.config.population_size
        ):
            best_id: Optional[PromptConfigurationId] = None
            best_score = float("-inf")
            for cand_id in self.prompt_configurations_by_id.keys():
                mean_score = self._mean_minibatch_score(cand_id)
                if mean_score > best_score:
                    best_score = mean_score
                    best_id = cand_id

            worst_id: Optional[PromptConfigurationId] = None
            worst_score = float("inf")
            for cand_id in self.prompt_configurations_by_id.keys():
                if cand_id == best_id:
                    continue
                mean_score = self._mean_minibatch_score(cand_id)
                if mean_score < worst_score:
                    worst_score = mean_score
                    worst_id = cand_id

            if worst_id is None or worst_id == best_id:
                break

            # Prune the chosen worst candidate from all bookkeeping tables.
            self.prompt_configurations_by_id.pop(worst_id, None)
            self.parents_by_id.pop(worst_id, None)
            self._minibatch_score_sums.pop(worst_id, None)
            self._minibatch_score_counts.pop(worst_id, None)
            self.pareto_score_table.pop(worst_id, None)

    def _record_minibatch_score(
        self,
        prompt_configuration_id: PromptConfigurationId,
        score: float,
    ) -> None:
        self._minibatch_score_sums[prompt_configuration_id] = (
            self._minibatch_score_sums.get(prompt_configuration_id, 0.0)
            + float(score)
        )
        self._minibatch_score_counts[prompt_configuration_id] = (
            self._minibatch_score_counts.get(prompt_configuration_id, 0) + 1
        )

    def _mean_minibatch_score(
        self,
        prompt_configuration_id: PromptConfigurationId,
    ) -> float:
        total = self._minibatch_score_sums.get(prompt_configuration_id, 0.0)
        count = self._minibatch_score_counts.get(prompt_configuration_id, 0)
        if count <= 0:
            # Use a sentinel that will not dominate selection if a scored
            # candidate exists. Root is seeded explicitly in the first iteration.
            return float("-inf")
        return total / count

    def _best_by_minibatch(self) -> PromptConfiguration:
        """
        Return the candidate with the highest mean minibatch score.
        """
        if not self.prompt_configurations_by_id:
            raise DeepEvalError(
                "COPRORunner has no prompt configurations; this should not happen."
            )

        best_id: Optional[PromptConfigurationId] = None
        best_score = float("-inf")

        for cand_id in self.prompt_configurations_by_id.keys():
            mean_score = self._mean_minibatch_score(cand_id)
            if mean_score > best_score:
                best_score = mean_score
                best_id = cand_id

        if best_id is None:
            # Fallback to the first candidate if all means are -inf.
            best_id = next(iter(self.prompt_configurations_by_id.keys()))

        return self.prompt_configurations_by_id[best_id]

    def _best_by_aggregate(self) -> PromptConfiguration:
        """
        Return the best candidate based on full-eval scores.

        If no full evaluation scores are available (should be rare, but possible if
        full_eval_every is very large and the loop exits early), fall back to
        best-by-minibatch.
        """
        if not self.pareto_score_table:
            return self._best_by_minibatch()

        totals = {
            prompt_configuration_id: self.aggregate_instances(vector)
            for prompt_configuration_id, vector in self.pareto_score_table.items()
        }

        best_ids: List[PromptConfigurationId] = []
        best_val = float("-inf")

        for cand_id, aggregate in totals.items():
            if aggregate > best_val + 1e-12:
                best_val = aggregate
                best_ids = [cand_id]
            elif abs(aggregate - best_val) <= 1e-12:
                best_ids.append(cand_id)

        chosen_id = self.random_state.choice(best_ids)
        return self.prompt_configurations_by_id[chosen_id]

    def _select_candidate(self) -> PromptConfiguration:
        """
        Epsilon-greedy candidate selection:

        - With probability ``exploration_probability``, pick a random candidate.
        - Otherwise, pick the candidate with the highest mean minibatch score.
        """
        if not self.prompt_configurations_by_id:
            raise DeepEvalError(
                "COPRORunner has no prompt configurations to select from."
            )

        candidate_ids = list(self.prompt_configurations_by_id.keys())
        if not candidate_ids:
            raise DeepEvalError(
                "COPRORunner has an empty candidate pool; this should not happen."
            )

        eps = float(self.config.exploration_probability)
        if eps > 0.0 and self.random_state.random() < eps:
            chosen_id = self.random_state.choice(candidate_ids)
        else:
            chosen_id = self._best_by_minibatch().id

        return self.prompt_configurations_by_id[chosen_id]

    def _draw_minibatch(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Union[List["Golden"], List["ConversationalGolden"]]:
        """
        Determine effective minibatch size from COPROConfig, bounded by the
        available goldens, and sample with replacement.
        """
        n = len(goldens)
        if n <= 0:
            return []

        if self.config.minibatch_size is not None:
            size = self.config.minibatch_size
        else:
            dynamic = max(1, int(round(n * self.config.minibatch_ratio)))
            size = max(
                self.config.minibatch_min_size,
                min(dynamic, self.config.minibatch_max_size),
            )

        size = max(1, min(size, n))

        return [goldens[self.random_state.randrange(0, n)] for _ in range(size)]

    async def _a_full_evaluate_best(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> None:
        if not self.prompt_configurations_by_id:
            return

        best = self._best_by_minibatch()
        if best.id in self.pareto_score_table:
            return

        scores = await self.scoring_adapter.a_score_on_pareto(best, goldens)
        self.pareto_score_table[best.id] = scores

    def _full_evaluate_best(
        self,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> None:
        if not self.prompt_configurations_by_id:
            return

        best = self._best_by_minibatch()
        if best.id in self.pareto_score_table:
            return

        scores = self.scoring_adapter.score_on_pareto(best, goldens)
        self.pareto_score_table[best.id] = scores

    async def _a_generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        feedback_text: str,
    ) -> Optional[Prompt]:
        try:
            old_prompt = parent_prompt_configuration.prompts[selected_module_id]
        except KeyError as exc:
            raise DeepEvalError(
                "COPRORunner expected a prompt for module_id "
                f"{selected_module_id!r} but none was found in the "
                "current prompt configuration."
            ) from exc

        new_prompt = await self._rewriter.a_rewrite(
            model_callback=self.model_callback,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )

        if old_prompt.type != new_prompt.type or self._prompts_equivalent(
            old_prompt, new_prompt
        ):
            # Don't accept if new prompt is the same as parent, or if type changed.
            return None
        return new_prompt

    def _generate_child_prompt(
        self,
        selected_module_id: ModuleId,
        parent_prompt_configuration: PromptConfiguration,
        feedback_text: str,
    ) -> Optional[Prompt]:
        try:
            old_prompt = parent_prompt_configuration.prompts[selected_module_id]
        except KeyError as exc:
            # This should never happen in normal operation.
            raise DeepEvalError(
                "COPRORunner expected a prompt for module_id "
                f"{selected_module_id!r} but none was found in the "
                "current prompt configuration."
            ) from exc

        new_prompt = self._rewriter.rewrite(
            model_callback=self.model_callback,
            module_id=selected_module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )

        if old_prompt.type != new_prompt.type or self._prompts_equivalent(
            old_prompt, new_prompt
        ):
            # Don't accept if new prompt is the same as parent, or if type changed.
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

    def _update_progress(
        self,
        total_iterations: int,
        iteration: int,
        remaining_iterations: int,
        elapsed: float,
    ) -> None:
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
        self,
        total_iterations: int,
        iteration: int,
        exc: Exception,
    ) -> None:
        # Report a user-facing error event.
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
        copro_iteration: Callable[[], bool],
    ) -> None:
        total_iterations = self.config.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(
            total_iterations, iteration, remaining_iterations, 0.0
        )
        while remaining_iterations > 0:
            iteration += 1
            start_time = time.perf_counter()
            try:
                ok = copro_iteration()
            except Exception as exc:
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
        a_copro_iteration: Callable[[], Awaitable[bool]],
    ) -> None:
        total_iterations = self.config.iterations
        remaining_iterations = total_iterations
        iteration = 0
        self._update_progress(
            total_iterations, iteration, remaining_iterations, 0.0
        )
        while remaining_iterations > 0:
            iteration += 1
            start_time = time.perf_counter()
            try:
                ok = await a_copro_iteration()
            except Exception as exc:
                self._update_error(total_iterations, iteration, exc)
                break
            elapsed = time.perf_counter() - start_time
            if not ok:
                break
            remaining_iterations -= 1
            self._update_progress(
                total_iterations, iteration, remaining_iterations, elapsed
            )
