from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple, Union

from rich import box
from rich.table import Table

from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.metrics.utils import copy_metrics
from deepeval.optimizer.algorithms.copro.proposer import COPROProposer
from deepeval.optimizer.algorithms.base import BaseAlgorithm
from deepeval.optimizer.scorer.utils import (
    _a_measure_no_indicator,
    _measure_no_indicator,
)
from deepeval.optimizer.types import (
    AcceptedIteration,
    IterationLogEntry,
    ModuleId,
    OptimizationReport,
    PromptConfiguration,
    RunnerStatusCallback,
    RunnerStatusType,
    ScoreTable,
)
from deepeval.optimizer.utils import build_prompt_config_snapshots
from deepeval.prompt.prompt import Prompt


class COPRO(BaseAlgorithm):
    """
    COPRO Optimizer (Lite Version - Single Module).
    Uses Informed Coordinate Ascent to iteratively refine instructions based on historical scores and metric feedback.
    """

    name = "COPRO"
    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        depth: int = 4,
        breadth: int = 7,
        minibatch_size: int = 25,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.minibatch_size = minibatch_size
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
        self.proposer = COPROProposer(
            optimizer_model=self.optimizer_model,
            random_state=self.random_state,
        )

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

    def _extract_optimized_set(self) -> Optional[str]:
        true_best_id: Optional[str] = None
        true_best_score = float("-inf")
        for cid, scores in self.pareto_score_table.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > true_best_score:
                true_best_score = avg_score
                true_best_id = cid
        return true_best_id

    def _evaluate_candidate(
        self, config: PromptConfiguration, minibatch: List
    ) -> Tuple[float, str]:
        scores = []
        failure_feedbacks = []

        for golden in minibatch:
            actual = self.scorer.generate(config.prompts, golden)
            test_case = self.scorer._golden_to_test_case(golden, actual)

            metrics = copy_metrics(self.scorer.metrics)
            for metric in metrics:
                _measure_no_indicator(metric, test_case)

            avg_score = (
                sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
            )
            scores.append(avg_score)

            if avg_score < 1.0 and len(failure_feedbacks) < 3:
                failure_feedbacks.append(
                    self.scorer._build_evaluation_results_block(
                        golden, actual, metrics
                    )
                )

        final_score = sum(scores) / len(scores) if scores else 0.0
        feedback_str = (
            "\n---\n".join(failure_feedbacks)
            if failure_feedbacks
            else "All metrics passed perfectly."
        )
        return final_score, feedback_str

    async def _a_evaluate_candidate(
        self, config: PromptConfiguration, minibatch: List
    ) -> Tuple[float, str]:
        async def process_one(golden):
            actual = await self.scorer.a_generate(config.prompts, golden)
            test_case = self.scorer._golden_to_test_case(golden, actual)
            metrics = copy_metrics(self.scorer.metrics)
            for metric in metrics:
                await _a_measure_no_indicator(metric, test_case)

            avg_score = (
                sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
            )
            feedback = (
                self.scorer._build_evaluation_results_block(
                    golden, actual, metrics
                )
                if avg_score < 1.0
                else None
            )
            return avg_score, feedback

        tasks = [process_one(g) for g in minibatch]
        results = await asyncio.gather(*tasks)

        scores = [res[0] for res in results]
        feedbacks = [res[1] for res in results if res[1] is not None]

        final_score = sum(scores) / len(scores) if scores else 0.0
        feedback_str = (
            "\n---\n".join(feedbacks[:3])
            if feedbacks
            else "All metrics passed perfectly."
        )
        return final_score, feedback_str

    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, OptimizationReport]:
        self.optimization_id = str(uuid.uuid4())
        self._init_components()
        self._iteration_log = []

        self._update_step(
            f"Bootstrapping {self.breadth} zero-shot variations..."
        )
        candidates = self.proposer.propose_bootstrap(prompt, self.breadth)
        candidates.insert(0, prompt)

        global_best_score = float("-inf")
        global_best_id: Optional[str] = None
        accepted_iterations: List[AcceptedIteration] = []
        history_log: List[Tuple[Prompt, float, str]] = []

        for d in range(self.depth):
            depth_start = time.time()
            self._update_trial_progress(d + 1, self.depth)
            self._update_step(
                f"Depth {d + 1}/{self.depth}: Evaluating {len(candidates)} candidates on minibatch..."
            )

            minibatch = self._sample_minibatch(goldens)
            batch_results = []

            for c in candidates:
                config = PromptConfiguration.new(
                    prompts={self.SINGLE_MODULE_ID: c}
                )
                self.prompt_configurations_by_id[config.id] = config

                score, feedback = self._evaluate_candidate(config, minibatch)
                batch_results.append((c, config, score, feedback))

            batch_results.sort(key=lambda x: x[2], reverse=True)
            best_batch_c, best_batch_config, best_batch_score, _ = (
                batch_results[0]
            )

            for c, _, score, feedback in batch_results[: self.breadth]:
                history_log.append((c, score, feedback))
            history_log.sort(key=lambda x: x[1], reverse=True)
            history_log = history_log[: self.breadth]

            self._iteration_log.append(
                IterationLogEntry(
                    iteration=d + 1,
                    outcome="evaluated",
                    before=(
                        global_best_score
                        if global_best_score != float("-inf")
                        else 0.0
                    ),
                    after=best_batch_score,
                    reason=f"Best Minibatch Candidate ID: {best_batch_config.id[:8]}",
                    elapsed=time.time() - depth_start,
                )
            )

            self._update_step(
                f"Depth {d + 1}/{self.depth}: Running full dataset validation on best candidate..."
            )
            full_scores = self.scorer.score_pareto(best_batch_config, goldens)
            avg_full_score = sum(full_scores) / len(full_scores)
            self.pareto_score_table[best_batch_config.id] = full_scores

            if avg_full_score > global_best_score:
                if global_best_id is not None:
                    accepted_iterations.append(
                        AcceptedIteration(
                            parent=global_best_id,
                            child=best_batch_config.id,
                            module=self.SINGLE_MODULE_ID,
                            before=global_best_score,
                            after=avg_full_score,
                        )
                    )
                    self.parents_by_id[best_batch_config.id] = global_best_id
                else:
                    self.parents_by_id.setdefault(best_batch_config.id, None)

                global_best_score = avg_full_score
                global_best_id = best_batch_config.id

            if d < self.depth - 1:
                self._update_step(
                    f"Depth {d + 1}/{self.depth}: Analyzing history and proposing next batch..."
                )
                candidates = self.proposer.propose_from_history(
                    best_batch_c, history_log, self.breadth
                )
                if not candidates:
                    candidates = [best_batch_c]

        true_best_id = self._extract_optimized_set()
        final_id = true_best_id if true_best_id else global_best_id
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

        self._update_step(f"Generating {self.breadth} variations...")
        candidates = await self.proposer.a_propose_bootstrap(
            prompt, self.breadth
        )
        candidates.insert(0, prompt)

        global_best_score = float("-inf")
        global_best_id: Optional[str] = None
        accepted_iterations: List[AcceptedIteration] = []
        history_log: List[Tuple[Prompt, float, str]] = []

        for d in range(self.depth):
            depth_start = time.time()
            self._update_trial_progress(d + 1, self.depth)
            self._update_step(
                f"Depth {d + 1}/{self.depth}: Evaluating {len(candidates)} candidates on minibatch concurrently..."
            )

            minibatch = self._sample_minibatch(goldens)
            batch_results = []
            configs = []

            for c in candidates:
                config = PromptConfiguration.new(
                    prompts={self.SINGLE_MODULE_ID: c}
                )
                self.prompt_configurations_by_id[config.id] = config
                configs.append(config)

            tasks = [
                self._a_evaluate_candidate(conf, minibatch) for conf in configs
            ]
            results = await asyncio.gather(*tasks)

            for c, conf, res in zip(candidates, configs, results):
                score, feedback = res
                batch_results.append((c, conf, score, feedback))

            batch_results.sort(key=lambda x: x[2], reverse=True)
            best_batch_c, best_batch_config, best_batch_score, _ = (
                batch_results[0]
            )

            for c, _, score, feedback in batch_results[: self.breadth]:
                history_log.append((c, score, feedback))
            history_log.sort(key=lambda x: x[1], reverse=True)
            history_log = history_log[: self.breadth]

            self._iteration_log.append(
                IterationLogEntry(
                    iteration=d + 1,
                    outcome="evaluated",
                    before=(
                        global_best_score
                        if global_best_score != float("-inf")
                        else 0.0
                    ),
                    after=best_batch_score,
                    reason=f"Best Minibatch Candidate ID: {best_batch_config.id[:8]}",
                    elapsed=time.time() - depth_start,
                )
            )

            self._update_step(
                f"Depth {d + 1}/{self.depth}: Running full dataset validation on best candidate..."
            )
            full_scores = await self.scorer.a_score_pareto(
                best_batch_config, goldens
            )
            avg_full_score = sum(full_scores) / len(full_scores)
            self.pareto_score_table[best_batch_config.id] = full_scores

            if avg_full_score > global_best_score:
                if global_best_id is not None:
                    accepted_iterations.append(
                        AcceptedIteration(
                            parent=global_best_id,
                            child=best_batch_config.id,
                            module=self.SINGLE_MODULE_ID,
                            before=global_best_score,
                            after=avg_full_score,
                        )
                    )
                    self.parents_by_id[best_batch_config.id] = global_best_id
                else:
                    self.parents_by_id.setdefault(best_batch_config.id, None)

                global_best_score = avg_full_score
                global_best_id = best_batch_config.id

            if d < self.depth - 1:
                self._update_step(
                    f"Depth {d + 1}/{self.depth}: Analyzing history and proposing next batch..."
                )
                candidates = await self.proposer.a_propose_from_history(
                    best_batch_c, history_log, self.breadth
                )
                if not candidates:
                    candidates = [best_batch_c]

        true_best_id = self._extract_optimized_set()
        final_id = true_best_id if true_best_id else global_best_id
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
            title=f"📈 [{_PURPLE}]{self.name}[/] Coordinate Ascent (Minibatch Trials)",
            box=box.ROUNDED,
            border_style=_PURPLE,
            header_style=f"bold {_PURPLE}",
            show_lines=True,
            expand=True,
        )
        iter_table.add_column(
            "Depth", style="bold white", justify="right", no_wrap=True
        )
        iter_table.add_column("Status", justify="center", no_wrap=True)
        iter_table.add_column("Best Prior", justify="right", no_wrap=True)
        iter_table.add_column("Batch Top Score", justify="right", no_wrap=True)
        iter_table.add_column("Δ to Best", justify="right", no_wrap=True)
        iter_table.add_column("Note", style=f"{_DIM}", no_wrap=False)
        iter_table.add_column("Time", justify="right", no_wrap=True)

        running_max = float("-inf")

        for entry in iteration_log:
            i = str(entry.iteration)
            score = entry.after
            reason = entry.reason
            elapsed = entry.elapsed

            best_prior = running_max if running_max != float("-inf") else 0.0
            delta = score - best_prior

            if score > running_max:
                status_cell = f"[{_GREEN}]▲ Ascended[/]"
                color = "white"
                sign = "+" if delta >= 0 else ""
                running_max = score
            else:
                status_cell = f"[{_DIM}]◆ Explored[/]"
                color = _DIM
                sign = "+" if delta >= 0 else ""

            best_prior_cell = f"{best_prior:.4f}"
            score_cell = (
                f"[bold {color}]{score:.4f}[/]"
                if score >= running_max
                else f"[{color}]{score:.4f}[/]"
            )
            delta_cell = f"[{color}]{sign}{delta:.4f}[/]"
            time_cell = f"[{_DIM}]{elapsed:.2f}s[/]"

            iter_table.add_row(
                i,
                status_cell,
                best_prior_cell,
                score_cell,
                delta_cell,
                reason,
                time_cell,
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
            pareto_table.add_column("Role", justify="center", no_wrap=True)
            pareto_table.add_column(
                "Scores Array", justify="center", no_wrap=False
            )
            pareto_table.add_column(
                "True Avg Score", justify="right", no_wrap=True
            )

            best_id = report.best_id

            for cid, scores in report.pareto_scores.items():
                is_best = cid == best_id
                role = f"[{_DIM}]candidate[/]"

                short_id = cid[:8] + "…"
                if is_best:
                    short_id = f"[bold white]{short_id} ★[/]"

                if len(scores) > 6:
                    score_strs = (
                        [f"{s:.3f}" for s in scores[:3]]
                        + ["..."]
                        + [f"{s:.3f}" for s in scores[-3:]]
                    )
                else:
                    score_strs = [f"{s:.3f}" for s in scores]
                scores_cell = f"[{_DIM}][{', '.join(score_strs)}][/]"

                agg = sum(scores) / len(scores) if scores else 0.0
                agg_color = "white" if is_best else _DIM
                agg_cell = (
                    f"[bold {agg_color}]{agg:.4f}[/]"
                    if is_best
                    else f"[{agg_color}]{agg:.4f}[/]"
                )

                pareto_table.add_row(short_id, role, scores_cell, agg_cell)

            tables.append(pareto_table)

        return tables
