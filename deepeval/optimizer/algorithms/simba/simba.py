from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple, Union

from rich import box
from rich.table import Table

from deepeval.dataset.golden import ConversationalGolden, Golden
from deepeval.metrics.utils import copy_metrics
from deepeval.optimizer.algorithms.base import BaseAlgorithm
from deepeval.optimizer.scorer.utils import (
    _a_measure_no_indicator,
    _measure_no_indicator,
)
from deepeval.optimizer.types import (
    AcceptedIteration,
    IterationLogEntry,
    ModuleId,
    SimbaTraceRecord,
    SimbaVarianceBucket,
    OptimizationReport,
    PromptConfiguration,
    RunnerStatusCallback,
    RunnerStatusType,
    ScoreTable,
)
from deepeval.optimizer.utils import build_prompt_config_snapshots
from deepeval.prompt.prompt import Prompt

from .proposer import SIMBAProposer


class SIMBA(BaseAlgorithm):

    name = "SIMBA"
    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        iterations: int = 8,
        minibatch_size: int = 15,
        num_candidates: int = 4,
        num_samples: int = 3,
        minibatch_full_eval_steps: int = 4,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        super().__init__()
        self.iterations = iterations
        self.minibatch_size = minibatch_size
        self.num_candidates = num_candidates
        self.num_samples = num_samples
        self.minibatch_full_eval_steps = minibatch_full_eval_steps
        self.pareto_score_table: ScoreTable = {}
        self.parents_by_id: Dict[str, Optional[str]] = {}
        self.prompt_configurations_by_id: Dict[str, PromptConfiguration] = {}
        self.step_callback: Optional[Callable[[str], None]] = None
        self.status_callback: Optional[RunnerStatusCallback] = None
        self.optimization_id: str = ""
        self._iteration_log: List[IterationLogEntry] = []

        if isinstance(random_state, int):
            self.seed = random_state
            self.random_state = random.Random(random_state)
        else:
            self.seed = random.randint(0, 999999)
            self.random_state = random_state or random.Random(self.seed)

    def _init_components(self) -> None:
        self.proposer = SIMBAProposer(optimizer_model=self.optimizer_model)

    def _sample_minibatch(self, goldens: List) -> List:
        if len(goldens) <= self.minibatch_size:
            return goldens
        return self.random_state.sample(goldens, self.minibatch_size)

    def _update_step(self, message: str) -> None:
        if self.step_callback is not None:
            self.step_callback(message)

    def _update_trial_progress(self, step: int, total: int) -> None:
        if self.status_callback is not None:
            self.status_callback(
                RunnerStatusType.PROGRESS,
                detail="",
                step_index=step,
                total_steps=total,
            )

    @staticmethod
    def _golden_expected_text(
        golden: Union[Golden, ConversationalGolden],
    ) -> Optional[str]:
        if isinstance(golden, Golden):
            return golden.expected_output
        return golden.expected_outcome

    def _extract_inputs(
        self, golden: Union[Golden, ConversationalGolden]
    ) -> str:
        if isinstance(golden, Golden):
            return golden.input
        return "\n".join(
            [t.content for t in (golden.turns or []) if t.role == "user"]
        )

    def _execute_trace(
        self,
        config: PromptConfiguration,
        golden: Union[Golden, ConversationalGolden],
    ) -> SimbaTraceRecord:
        actual = self.scorer.generate(config.prompts, golden)
        test_case = self.scorer._golden_to_test_case(golden, actual)

        metrics = copy_metrics(self.scorer.metrics)
        score_sum = 0
        reasons = []
        for metric in metrics:
            _measure_no_indicator(metric, test_case)
            score_sum += metric.score
            reasons.append(
                f"- {metric.__class__.__name__} ({metric.score}): {metric.reason}"
            )

        avg_score = score_sum / len(metrics) if metrics else 0.0
        return SimbaTraceRecord(
            output=actual,
            score=avg_score,
            feedback="\n".join(reasons),
        )

    async def _a_execute_trace(
        self,
        config: PromptConfiguration,
        golden: Union[Golden, ConversationalGolden],
    ) -> SimbaTraceRecord:
        actual = await self.scorer.a_generate(config.prompts, golden)
        test_case = self.scorer._golden_to_test_case(golden, actual)

        metrics = copy_metrics(self.scorer.metrics)
        score_sum = 0
        reasons = []
        for metric in metrics:
            await _a_measure_no_indicator(metric, test_case)
            score_sum += metric.score
            reasons.append(
                f"- {metric.__class__.__name__} ({metric.score}): {metric.reason}"
            )

        avg_score = score_sum / len(metrics) if metrics else 0.0
        return SimbaTraceRecord(
            output=actual,
            score=avg_score,
            feedback="\n".join(reasons),
        )

    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, OptimizationReport]:
        self.optimization_id = str(uuid.uuid4())
        self._init_components()
        self._iteration_log = []

        root_config = PromptConfiguration.new(
            prompts={self.SINGLE_MODULE_ID: prompt}
        )
        self.prompt_configurations_by_id[root_config.id] = root_config
        self.parents_by_id[root_config.id] = None

        current_best_config = root_config
        global_best_score = float("-inf")
        accepted_iterations: List[AcceptedIteration] = []

        for trial_idx in range(self.iterations):
            trial_start = time.time()
            self._update_trial_progress(trial_idx + 1, self.iterations)

            minibatch = self._sample_minibatch(goldens)

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Sampling trajectories for introspection..."
            )
            buckets: List[SimbaVarianceBucket] = []

            for golden in minibatch:
                traces_raw = [
                    self._execute_trace(current_best_config, golden)
                    for _ in range(self.num_samples)
                ]
                traces = sorted(traces_raw, key=lambda t: t.score, reverse=True)

                max_score = traces[0].score
                min_score = traces[-1].score
                avg_score = sum(t.score for t in traces) / len(traces)

                buckets.append(
                    SimbaVarianceBucket(
                        golden=golden,
                        traces=traces,
                        max_to_avg_gap=max_score - avg_score,
                        max_score=max_score,
                        min_score=min_score,
                    )
                )

            buckets.sort(
                key=lambda b: (b.max_to_avg_gap, -b.max_score), reverse=True
            )

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Introspecting hard examples..."
            )
            candidate_configs = []

            for bucket in buckets[: self.num_candidates]:
                golden = bucket.golden
                inputs = self._extract_inputs(golden)

                force_rule_strategy = False

                if bucket.max_to_avg_gap > 0:
                    good_trace = bucket.traces[0]
                    bad_trace = bucket.traces[-1]

                    if good_trace.score < 0.8:
                        expected = self._golden_expected_text(golden)
                        if expected:
                            good_trace = SimbaTraceRecord(
                                output=str(expected),
                                score=1.0,
                                feedback="This is the optimal, ground-truth expected output.",
                            )
                        else:
                            force_rule_strategy = True
                else:
                    if bucket.max_score >= 0.99:
                        continue

                    expected = self._golden_expected_text(golden)
                    if not expected:
                        continue

                    bad_trace = bucket.traces[0]
                    good_trace = SimbaTraceRecord(
                        output=str(expected),
                        score=1.0,
                        feedback="This is the optimal, ground-truth expected output.",
                    )

                if force_rule_strategy:
                    strategy = "rule"
                else:
                    strategy = self.random_state.choice(["rule", "demo"])

                try:
                    if strategy == "rule":
                        new_prompt = self.proposer.rewrite_from_introspection(
                            original_prompt=current_best_config.prompts[
                                self.SINGLE_MODULE_ID
                            ],
                            better_inputs=inputs,
                            better_outputs=str(good_trace.output),
                            better_score=good_trace.score,
                            better_feedback=good_trace.feedback,
                            worse_inputs=inputs,
                            worse_outputs=str(bad_trace.output),
                            worse_score=bad_trace.score,
                            worse_feedback=bad_trace.feedback,
                        )
                    else:
                        new_prompt = self.proposer.append_a_demo(
                            original_prompt=current_best_config.prompts[
                                self.SINGLE_MODULE_ID
                            ],
                            inputs=inputs,
                            outputs=str(good_trace.output),
                        )

                    config = PromptConfiguration.new(
                        prompts={self.SINGLE_MODULE_ID: new_prompt},
                        parent=current_best_config.id,
                    )
                    self.prompt_configurations_by_id[config.id] = config
                    candidate_configs.append(config)
                except Exception:
                    continue

            if not candidate_configs:
                self._iteration_log.append(
                    IterationLogEntry(
                        iteration=trial_idx + 1,
                        outcome="skipped",
                        before=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        after=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        reason="No introspectable variance or ground-truths found.",
                        elapsed=time.time() - trial_start,
                    )
                )
                continue

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Evaluating {len(candidate_configs)} candidates..."
            )
            batch_results = []

            for config in candidate_configs:
                score = self.scorer.score_minibatch(config, minibatch)
                batch_results.append((config, score))

            batch_results.sort(key=lambda x: x[1], reverse=True)
            best_batch_config, best_batch_score = batch_results[0]

            if (
                (trial_idx + 1) % self.minibatch_full_eval_steps == 0
                or trial_idx == self.iterations - 1
            ):
                self._update_step(
                    "Running full validation on current best configuration..."
                )

                full_scores = self.scorer.score_pareto(
                    best_batch_config, goldens
                )
                avg_full_score = sum(full_scores) / len(full_scores)
                self.pareto_score_table[best_batch_config.id] = full_scores

                if avg_full_score > global_best_score:
                    accepted_iterations.append(
                        AcceptedIteration(
                            parent=current_best_config.id,
                            child=best_batch_config.id,
                            module=self.SINGLE_MODULE_ID,
                            before=(
                                global_best_score
                                if global_best_score != float("-inf")
                                else 0.0
                            ),
                            after=avg_full_score,
                        )
                    )
                    self.parents_by_id[best_batch_config.id] = (
                        current_best_config.id
                    )
                    global_best_score = avg_full_score
                    current_best_config = best_batch_config
                    outcome = "accepted"
                else:
                    outcome = "rejected"

                self._iteration_log.append(
                    IterationLogEntry(
                        iteration=trial_idx + 1,
                        outcome=outcome,
                        before=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        after=avg_full_score,
                        reason="Evaluated on full dataset.",
                        elapsed=time.time() - trial_start,
                    )
                )

        true_best_id: Optional[str] = None
        true_best_score = float("-inf")
        for cid, scores in self.pareto_score_table.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > true_best_score:
                true_best_score = avg_score
                true_best_id = cid

        final_id = true_best_id if true_best_id else current_best_config.id
        best_config = self.prompt_configurations_by_id[final_id]

        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best_config.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=build_prompt_config_snapshots(
                self.prompt_configurations_by_id
            ),
        )

        return best_config.prompts[self.SINGLE_MODULE_ID], report

    async def a_execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, OptimizationReport]:
        self.optimization_id = str(uuid.uuid4())
        self._init_components()
        self._iteration_log = []

        root_config = PromptConfiguration.new(
            prompts={self.SINGLE_MODULE_ID: prompt}
        )
        self.prompt_configurations_by_id[root_config.id] = root_config
        self.parents_by_id[root_config.id] = None

        current_best_config = root_config
        global_best_score = float("-inf")
        accepted_iterations: List[AcceptedIteration] = []

        for trial_idx in range(self.iterations):
            trial_start = time.time()
            self._update_trial_progress(trial_idx + 1, self.iterations)

            minibatch = self._sample_minibatch(goldens)

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Sampling trajectories for introspection..."
            )
            buckets: List[SimbaVarianceBucket] = []

            for golden in minibatch:
                tasks = [
                    self._a_execute_trace(current_best_config, golden)
                    for _ in range(self.num_samples)
                ]
                traces = await asyncio.gather(*tasks)
                traces = sorted(traces, key=lambda t: t.score, reverse=True)

                max_score = traces[0].score
                min_score = traces[-1].score
                avg_score = sum(t.score for t in traces) / len(traces)

                buckets.append(
                    SimbaVarianceBucket(
                        golden=golden,
                        traces=list(traces),
                        max_to_avg_gap=max_score - avg_score,
                        max_score=max_score,
                        min_score=min_score,
                    )
                )

            buckets.sort(
                key=lambda b: (b.max_to_avg_gap, -b.max_score), reverse=True
            )

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Introspecting hard examples..."
            )
            candidate_configs = []

            async def process_bucket(
                bucket: SimbaVarianceBucket,
            ) -> Optional[PromptConfiguration]:
                golden = bucket.golden
                inputs = self._extract_inputs(golden)

                force_rule_strategy = False

                if bucket.max_to_avg_gap > 0:
                    good_trace = bucket.traces[0]
                    bad_trace = bucket.traces[-1]

                    if good_trace.score < 0.8:
                        expected = self._golden_expected_text(golden)
                        if expected:
                            good_trace = SimbaTraceRecord(
                                output=str(expected),
                                score=1.0,
                                feedback="This is the optimal, ground-truth expected output.",
                            )
                        else:
                            force_rule_strategy = True
                else:
                    if bucket.max_score >= 0.99:
                        return None

                    expected = self._golden_expected_text(golden)
                    if not expected:
                        return None

                    bad_trace = bucket.traces[0]
                    good_trace = SimbaTraceRecord(
                        output=str(expected),
                        score=1.0,
                        feedback="This is the optimal, ground-truth expected output.",
                    )

                if force_rule_strategy:
                    strategy = "rule"
                else:
                    strategy = self.random_state.choice(["rule", "demo"])

                try:
                    if strategy == "rule":
                        new_prompt = (
                            await self.proposer.a_rewrite_from_introspection(
                                original_prompt=current_best_config.prompts[
                                    self.SINGLE_MODULE_ID
                                ],
                                better_inputs=inputs,
                                better_outputs=str(good_trace.output),
                                better_score=good_trace.score,
                                better_feedback=good_trace.feedback,
                                worse_inputs=inputs,
                                worse_outputs=str(bad_trace.output),
                                worse_score=bad_trace.score,
                                worse_feedback=bad_trace.feedback,
                            )
                        )
                    else:
                        new_prompt = self.proposer.append_a_demo(
                            original_prompt=current_best_config.prompts[
                                self.SINGLE_MODULE_ID
                            ],
                            inputs=inputs,
                            outputs=str(good_trace.output),
                        )

                    return PromptConfiguration.new(
                        prompts={self.SINGLE_MODULE_ID: new_prompt},
                        parent=current_best_config.id,
                    )
                except Exception:
                    return None

            pb_tasks = [
                process_bucket(b) for b in buckets[: self.num_candidates]
            ]
            results = await asyncio.gather(*pb_tasks)

            for res in results:
                if res:
                    self.prompt_configurations_by_id[res.id] = res
                    candidate_configs.append(res)

            if not candidate_configs:
                self._iteration_log.append(
                    IterationLogEntry(
                        iteration=trial_idx + 1,
                        outcome="skipped",
                        before=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        after=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        reason="No introspectable variance or ground-truths found.",
                        elapsed=time.time() - trial_start,
                    )
                )
                continue

            self._update_step(
                f"Iter {trial_idx + 1}/{self.iterations}: Evaluating {len(candidate_configs)} candidates..."
            )

            eval_tasks = [
                self.scorer.a_score_minibatch(config, minibatch)
                for config in candidate_configs
            ]
            scores = await asyncio.gather(*eval_tasks)

            batch_results = list(zip(candidate_configs, scores))
            batch_results.sort(key=lambda x: x[1], reverse=True)
            best_batch_config, best_batch_score = batch_results[0]

            if (
                (trial_idx + 1) % self.minibatch_full_eval_steps == 0
                or trial_idx == self.iterations - 1
            ):
                self._update_step(
                    "Running full validation on current best configuration..."
                )

                full_scores = await self.scorer.a_score_pareto(
                    best_batch_config, goldens
                )
                avg_full_score = sum(full_scores) / len(full_scores)
                self.pareto_score_table[best_batch_config.id] = full_scores

                if avg_full_score > global_best_score:
                    accepted_iterations.append(
                        AcceptedIteration(
                            parent=current_best_config.id,
                            child=best_batch_config.id,
                            module=self.SINGLE_MODULE_ID,
                            before=(
                                global_best_score
                                if global_best_score != float("-inf")
                                else 0.0
                            ),
                            after=avg_full_score,
                        )
                    )
                    self.parents_by_id[best_batch_config.id] = (
                        current_best_config.id
                    )
                    global_best_score = avg_full_score
                    current_best_config = best_batch_config
                    outcome = "accepted"
                else:
                    outcome = "rejected"

                self._iteration_log.append(
                    IterationLogEntry(
                        iteration=trial_idx + 1,
                        outcome=outcome,
                        before=(
                            global_best_score
                            if global_best_score != float("-inf")
                            else 0.0
                        ),
                        after=avg_full_score,
                        reason="Evaluated on full dataset.",
                        elapsed=time.time() - trial_start,
                    )
                )

        true_best_id: Optional[str] = None
        true_best_score = float("-inf")
        for cid, scores in self.pareto_score_table.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > true_best_score:
                true_best_score = avg_score
                true_best_id = cid

        final_id = true_best_id if true_best_id else current_best_config.id
        best_config = self.prompt_configurations_by_id[final_id]

        report = OptimizationReport(
            optimization_id=self.optimization_id,
            best_id=best_config.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=build_prompt_config_snapshots(
                self.prompt_configurations_by_id
            ),
        )

        return best_config.prompts[self.SINGLE_MODULE_ID], report

    def generate_summary_table(self, report: OptimizationReport) -> List[Table]:
        _PURPLE = "rgb(106,0,255)"
        _GREEN = "rgb(25,227,160)"
        _DIM = "rgb(55,65,81)"

        tables = []
        iteration_log = self._iteration_log

        iter_table = Table(
            title=f"🧠 [{_PURPLE}]{self.name}[/] Introspective Ascent",
            box=box.ROUNDED,
            border_style=_PURPLE,
            header_style=f"bold {_PURPLE}",
            show_lines=True,
            expand=True,
        )
        iter_table.add_column(
            "Iter", style="bold white", justify="right", no_wrap=True
        )
        iter_table.add_column("Status", justify="center", no_wrap=True)
        iter_table.add_column("Score Before", justify="right", no_wrap=True)
        iter_table.add_column("Score After", justify="right", no_wrap=True)
        iter_table.add_column("Note", style=f"{_DIM}", no_wrap=False)
        iter_table.add_column("Time", justify="right", no_wrap=True)

        for entry in iteration_log:
            i = str(entry.iteration)
            outcome = entry.outcome
            before = entry.before
            after = entry.after
            reason = entry.reason
            elapsed = entry.elapsed

            if outcome == "accepted":
                status_cell = f"[{_GREEN}]▲ Ascended[/]"
            elif outcome == "rejected":
                status_cell = f"[{_DIM}]◆ Explored[/]"
            else:
                status_cell = f"[{_DIM}]↷ Skipped[/]"

            before_cell = f"{before:.4f}"
            after_cell = (
                f"[bold white]{after:.4f}[/]"
                if outcome == "accepted"
                else f"[{_DIM}]{after:.4f}[/]"
            )
            time_cell = f"[{_DIM}]{elapsed:.2f}s[/]"

            iter_table.add_row(
                i, status_cell, before_cell, after_cell, reason, time_cell
            )

        tables.append(iter_table)

        if report and report.pareto_scores:
            pareto_table = Table(
                title=f"[{_PURPLE}]True Validation Archive (Full Dataset)[/]",
                box=box.HORIZONTALS,
                border_style=_PURPLE,
                header_style=f"bold {_PURPLE}",
                show_lines=True,
                expand=True,
            )
            pareto_table.add_column(
                "Config ID", style="white", justify="center", no_wrap=True
            )
            pareto_table.add_column(
                "Scores Array", justify="center", no_wrap=False
            )
            pareto_table.add_column(
                "True Avg Score", justify="right", no_wrap=True
            )

            best_id = report.best_id

            for cid, scores in report.pareto_scores.items():
                is_best = cid == best_id
                short_id = (
                    f"[bold white]{cid[:8]}… ★[/]" if is_best else f"{cid[:8]}…"
                )

                score_strs = [f"{s:.3f}" for s in scores]
                if len(score_strs) > 6:
                    score_strs = score_strs[:3] + ["..."] + score_strs[-3:]
                scores_cell = f"[{_DIM}][{', '.join(score_strs)}][/]"

                agg = sum(scores) / len(scores) if scores else 0.0
                agg_cell = (
                    f"[bold white]{agg:.4f}[/]"
                    if is_best
                    else f"[{_DIM}]{agg:.4f}[/]"
                )

                pareto_table.add_row(short_id, scores_cell, agg_cell)

            tables.append(pareto_table)

        return tables
