# MIPROv2 - Multiprompt Instruction PRoposal Optimizer Version 2
#
# This implementation follows the original MIPROv2 paper:
# https://arxiv.org/pdf/2406.11695
#
# Phase 1: Propose N diverse instructions and bootstrap M demo sets.
# Phase 2: Use Bayesian Optimization (Optuna TPE) to search the joint
#          categorical space of (Instruction, Demonstration Set) using
#          stochastic minibatch evaluation and periodic full evaluations.

from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Union, Optional
from rich.table import Table
from rich import box
import re

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None

from deepeval.errors import DeepEvalError
from deepeval.prompt.prompt import Prompt
from deepeval.optimizer.types import (
    AcceptedIterationDict,
    PromptConfiguration,
    ModuleId,
    ScoreTable,
    OptimizationReport,
    RunnerStatusType,
)
from deepeval.optimizer.algorithms.base import BaseAlgorithm
from deepeval.optimizer.utils import build_prompt_config_snapshots
from deepeval.optimizer.algorithms.miprov2.proposer.proposer import (
    InstructionProposer,
)
from deepeval.optimizer.algorithms.miprov2.bootstrapper import (
    DemonstrationBootstrapper,
    render_prompt_with_demonstrations,
)
from deepeval.dataset.golden import Golden, ConversationalGolden


class MIPROV2(BaseAlgorithm):
    """
    MIPROv2 Optimizer (Lite Version - Single Module).
    Uses Bayesian optimization over generated instructions and bootstrapped demos.
    """

    name = "MIPROv2"
    SINGLE_MODULE_ID: ModuleId = "__module__"

    def __init__(
        self,
        num_trials: int = 30,
        num_candidates: int = 10,
        max_bootstrapped_demonstrations: int = 4,
        max_labeled_demonstrations: int = 4,
        num_demonstration_sets: int = 5,
        minibatch_size: int = 25,
        minibatch_full_eval_steps: int = 10,
        random_state: Optional[Union[int, random.Random]] = None,
    ):
        super().__init__()
        if not OPTUNA_AVAILABLE:
            raise DeepEvalError(
                "MIPROv2 requires optuna. Please run `pip install optuna`."
            )

        self.num_trials = num_trials
        self.num_candidates = num_candidates
        self.max_bootstrapped_demonstrations = max_bootstrapped_demonstrations
        self.max_labeled_demonstrations = max_labeled_demonstrations
        self.num_demonstration_sets = num_demonstration_sets
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps

        # Internal State Tracking
        self.pareto_score_table: ScoreTable = {}
        self.parents_by_id: Dict[str, str] = {}
        self._config_cache: Dict[Tuple[int, int], PromptConfiguration] = {}
        self.prompt_configurations_by_id: Dict[str, PromptConfiguration] = {}

        self.candidates: List[Prompt] = []
        self.demo_sets = []

        if isinstance(random_state, int):
            self.seed = random_state
            self.random_state = random.Random(random_state)
        else:
            self.seed = random.randint(0, 999999)
            self.random_state = random_state or random.Random(self.seed)

    def _init_components(self) -> None:
        """Initialize the Proposer and Bootstrapper using the injected models."""
        self.proposer = InstructionProposer(
            optimizer_model=self.optimizer_model, random_state=self.random_state
        )
        self.bootstrapper = DemonstrationBootstrapper(
            scorer=self.scorer,
            max_bootstrapped_demonstrations=self.max_bootstrapped_demonstrations,
            max_labeled_demonstrations=self.max_labeled_demonstrations,
            num_demonstration_sets=self.num_demonstration_sets,
            random_state=self.random_state,
        )

    def _sample_minibatch(self, goldens: List) -> List:
        """Sample a stochastic minibatch for Optuna evaluation."""
        if len(goldens) <= self.minibatch_size:
            return goldens
        return self.random_state.sample(goldens, self.minibatch_size)

    def _build_config(
        self, instr_idx: int, demo_idx: int
    ) -> PromptConfiguration:
        """Stitch an instruction and demo set into a unified prompt configuration, using a cache to prevent ID leaks."""
        cache_key = (instr_idx, demo_idx)
        if hasattr(self, "_config_cache") and cache_key in self._config_cache:
            return self._config_cache[cache_key]

        base_prompt = self.candidates[instr_idx]
        demo_set = self.demo_sets[demo_idx]

        unified_prompt = render_prompt_with_demonstrations(
            base_prompt, demo_set
        )

        config = PromptConfiguration.new(
            prompts={self.SINGLE_MODULE_ID: unified_prompt}
        )
        self.prompt_configurations_by_id[config.id] = config

        if hasattr(self, "_config_cache"):
            self._config_cache[cache_key] = config

        return config

    def _update_step(self, message: str) -> None:
        """Updates the bottom text row (e.g., '⤷ Bootstrapping...')"""
        if getattr(self, "step_callback", None) is not None:
            self.step_callback(message)

    def _update_trial_progress(self, step: int, total: int) -> None:
        """Advances the main top progress bar."""
        if getattr(self, "status_callback", None) is not None:
            self.status_callback(
                RunnerStatusType.PROGRESS,
                detail="",
                step_index=step,
                total_steps=total,
            )

    ##################################################
    # Synchronous Execution
    ##################################################

    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, OptimizationReport]:
        import uuid

        self._init_components()
        self._iteration_log = []

        # Phase 1: Propose & Bootstrap
        self._update_step(
            f"Generating {self.num_candidates} diverse instructions..."
        )
        self.candidates = self.proposer.propose(
            prompt, goldens, self.num_candidates
        )

        self._update_step(
            f"Bootstrapping {self.num_demonstration_sets} verified demonstration sets..."
        )
        self.demo_sets = self.bootstrapper.bootstrap(prompt, goldens)

        # Phase 2: Bayesian Optimization
        self._update_step(
            "Initializing Tree-structured Parzen Estimator (TPE)..."
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.seed)
        )

        best_score = float("-inf")
        best_config_id = None
        accepted_iterations: List[AcceptedIterationDict] = []

        for trial_idx in range(self.num_trials):
            trial_start = time.time()
            self._update_trial_progress(trial_idx + 1, self.num_trials)
            self._update_step(
                f"Running Bayesian Trial {trial_idx + 1}/{self.num_trials}..."
            )

            trial = study.ask()
            instr_idx = trial.suggest_categorical(
                "instr_idx", list(range(len(self.candidates)))
            )
            demo_idx = trial.suggest_categorical(
                "demo_idx", list(range(len(self.demo_sets)))
            )

            config = self._build_config(instr_idx, demo_idx)
            minibatch = self._sample_minibatch(goldens)

            score = self.scorer.score_minibatch(config, minibatch)
            study.tell(trial, score)

            self._iteration_log.append(
                {
                    "iteration": trial_idx + 1,
                    "outcome": "accepted" if score > best_score else "rejected",
                    "before": (
                        best_score if best_score != float("-inf") else 0.0
                    ),
                    "after": score,
                    "reason": f"TPE Sample -> Instruction: {instr_idx}, DemoSet: {demo_idx}",
                    "elapsed": time.time() - trial_start,
                }
            )

            # Periodic Full Pareto Evaluation
            if (
                (trial_idx + 1) % self.minibatch_full_eval_steps == 0
                or trial_idx == self.num_trials - 1
            ):
                self._update_step(
                    f"Running full validation on current best configuration..."
                )
                best_trial = study.best_trial
                best_eval_config = self._build_config(
                    best_trial.params["instr_idx"],
                    best_trial.params["demo_idx"],
                )

                full_scores = self.scorer.score_pareto(
                    best_eval_config, goldens
                )
                avg_full_score = sum(full_scores) / len(full_scores)

                self.pareto_score_table[best_eval_config.id] = full_scores

                if avg_full_score > best_score:
                    if best_config_id is not None:
                        accepted_iterations.append(
                            AcceptedIterationDict(
                                parent=best_config_id,
                                child=best_eval_config.id,
                                module=self.SINGLE_MODULE_ID,
                                before=best_score,
                                after=avg_full_score,
                            )
                        )
                    best_score = avg_full_score
                    best_config_id = best_eval_config.id

        true_best_id = None
        true_best_score = float("-inf")
        for cid, scores in self.pareto_score_table.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > true_best_score:
                true_best_score = avg_score
                true_best_id = cid

        final_id = true_best_id if true_best_id else best_config_id
        best_config = self.prompt_configurations_by_id[final_id]

        report = OptimizationReport(
            optimization_id=getattr(self, "optimization_id", str(uuid.uuid4())),
            best_id=best_config.id,
            accepted_iterations=accepted_iterations,
            pareto_scores=self.pareto_score_table,
            parents=self.parents_by_id,
            prompt_configurations=build_prompt_config_snapshots(
                self.prompt_configurations_by_id
            ),
        )

        return best_config.prompts[self.SINGLE_MODULE_ID], report

    ##################################################
    # Asynchronous Execution
    ##################################################

    async def a_execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, OptimizationReport]:
        import uuid

        self._init_components()
        self._iteration_log = []

        self._update_step(
            f"Generating {self.num_candidates} diverse instructions..."
        )
        self.candidates = await self.proposer.a_propose(
            prompt, goldens, self.num_candidates
        )

        self._update_step(
            f"Bootstrapping {self.num_demonstration_sets} verified demonstration sets..."
        )
        self.demo_sets = await self.bootstrapper.a_bootstrap(prompt, goldens)

        self._update_step(
            "Initializing Tree-structured Parzen Estimator (TPE)..."
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.seed)
        )

        best_score = float("-inf")
        best_config_id = None
        accepted_iterations: List[AcceptedIterationDict] = []

        for trial_idx in range(self.num_trials):
            trial_start = time.time()
            self._update_trial_progress(trial_idx + 1, self.num_trials)
            self._update_step(
                f"Running Bayesian Trial {trial_idx + 1}/{self.num_trials}..."
            )

            trial = study.ask()
            instr_idx = trial.suggest_categorical(
                "instr_idx", list(range(len(self.candidates)))
            )
            demo_idx = trial.suggest_categorical(
                "demo_idx", list(range(len(self.demo_sets)))
            )

            config = self._build_config(instr_idx, demo_idx)
            minibatch = self._sample_minibatch(goldens)

            score = await self.scorer.a_score_minibatch(config, minibatch)
            study.tell(trial, score)

            self._iteration_log.append(
                {
                    "iteration": trial_idx + 1,
                    "outcome": "accepted" if score > best_score else "rejected",
                    "before": (
                        best_score if best_score != float("-inf") else 0.0
                    ),
                    "after": score,
                    "reason": f"TPE Sample -> Instruction: {instr_idx}, DemoSet: {demo_idx}",
                    "elapsed": time.time() - trial_start,
                }
            )

            if (
                (trial_idx + 1) % self.minibatch_full_eval_steps == 0
                or trial_idx == self.num_trials - 1
            ):
                self._update_step(
                    f"Running full validation on current best configuration..."
                )
                best_trial = study.best_trial
                best_eval_config = self._build_config(
                    best_trial.params["instr_idx"],
                    best_trial.params["demo_idx"],
                )

                full_scores = await self.scorer.a_score_pareto(
                    best_eval_config, goldens
                )
                avg_full_score = sum(full_scores) / len(full_scores)

                self.pareto_score_table[best_eval_config.id] = full_scores

                if avg_full_score > best_score:
                    if best_config_id is not None:
                        accepted_iterations.append(
                            AcceptedIterationDict(
                                parent=best_config_id,
                                child=best_eval_config.id,
                                module=self.SINGLE_MODULE_ID,
                                before=best_score,
                                after=avg_full_score,
                            )
                        )
                    best_score = avg_full_score
                    best_config_id = best_eval_config.id

        true_best_id = None
        true_best_score = float("-inf")
        for cid, scores in self.pareto_score_table.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > true_best_score:
                true_best_score = avg_score
                true_best_id = cid

        final_id = true_best_id if true_best_id else best_config_id
        best_config = self.prompt_configurations_by_id[final_id]

        report = OptimizationReport(
            optimization_id=getattr(self, "optimization_id", str(uuid.uuid4())),
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
        """Generates MIPROv2-specific Bayesian Search logs and Validation tables."""
        from rich.table import Table
        from rich import box

        _PURPLE = "rgb(106,0,255)"
        _GREEN = "rgb(25,227,160)"
        _DIM = "rgb(55,65,81)"

        tables = []
        iteration_log = getattr(self, "_iteration_log", [])

        # 1. Bayesian TPE Trial Table
        iter_table = Table(
            title=f"🔬 [{_PURPLE}]{self.name}[/] Bayesian Search (Stochastic Minibatches)",
            box=box.ROUNDED,
            border_style=_PURPLE,
            header_style=f"bold {_PURPLE}",
            show_lines=True,
            expand=True,
        )
        iter_table.add_column(
            "#", style="bold white", justify="right", no_wrap=True
        )
        iter_table.add_column("Status", justify="center", no_wrap=True)
        iter_table.add_column("Best Prior", justify="right", no_wrap=True)
        iter_table.add_column("Trial Score", justify="right", no_wrap=True)
        iter_table.add_column("Δ to Best", justify="right", no_wrap=True)
        iter_table.add_column("Note", style=f"{_DIM}", no_wrap=False)
        iter_table.add_column("Time", justify="right", no_wrap=True)

        running_max = float("-inf")

        for entry in iteration_log:
            i = str(entry["iteration"])
            score = entry.get("after", 0.0)
            reason = entry.get("reason", "")
            elapsed = entry.get("elapsed", 0.0)

            # Define the "Before" state as the highest score seen up to this point
            best_prior = running_max if running_max != float("-inf") else 0.0
            delta = score - best_prior

            # If it's a new high score, update the running max and mark it
            if score > running_max:
                status_cell = f"[{_GREEN}]🏆 New Best[/]"
                color = "white"
                sign = "+" if delta >= 0 else ""
                running_max = score
            else:
                status_cell = f"[{_DIM}]📊 Sampled[/]"
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

        # 2. Final Pareto archive table
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
