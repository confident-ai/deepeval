import sys
from contextlib import contextmanager
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.errors import DeepEvalError
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.metrics.utils import initialize_model
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.scorer import Scorer
from deepeval.optimizer.rewriter import Rewriter
from deepeval.optimizer.types import (
    ModelCallback,
    RunnerStatusType,
)
from deepeval.optimizer.utils import (
    validate_callback,
    validate_metrics,
)
from deepeval.optimizer.configs import (
    DisplayConfig,
    AsyncConfig,
)
from deepeval.prompt.prompt import Prompt
from deepeval.utils import get_or_create_event_loop
from deepeval.optimizer.algorithms import (
    GEPA,
    MIPROV2,
    COPRO,
    SIMBA,
)
from deepeval.optimizer.algorithms.configs import (
    GEPA_REWRITE_INSTRUCTION_MAX_CHARS,
    MIPROV2_REWRITE_INSTRUCTION_MAX_CHARS,
)


class PromptOptimizer:
    def __init__(
        self,
        model_callback: ModelCallback,
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
        optimizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        algorithm: Union[GEPA, MIPROV2, COPRO, SIMBA] = GEPA(),
        async_config: Optional[AsyncConfig] = AsyncConfig(),
        display_config: Optional[DisplayConfig] = DisplayConfig(),
    ):
        self.optimizer_model, self.using_native_model = initialize_model(
            optimizer_model
        )
        self.model_callback = validate_callback(
            component="PromptOptimizer",
            model_callback=model_callback,
        )
        self.metrics = validate_metrics(
            component="PromptOptimizer", metrics=metrics
        )

        self.async_config = async_config
        self.display_config = display_config
        self.algorithm = algorithm
        self.optimization_report = None
        self._configure_algorithm()

        # Internal state used only when a progress indicator is active.
        # Tuple is (Progress instance, task_id).
        self._progress_state: Optional[Tuple[Progress, int, int]] = None

    ##############
    # Public API #
    ##############

    def optimize(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Prompt:
        if self.async_config.run_async:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_optimize(prompt=prompt, goldens=goldens)
            )

        with self._progress_context():
            best_prompt, self.optimization_report = self.algorithm.execute(
                prompt=prompt, goldens=goldens
            )

        if self.display_config.show_indicator:
            self._print_summary_table()

        return best_prompt

    async def a_optimize(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Prompt:
        with self._progress_context():
            best_prompt, self.optimization_report = (
                await self.algorithm.a_execute(prompt=prompt, goldens=goldens)
            )

        if self.display_config.show_indicator:
            self._print_summary_table()

        return best_prompt

    ####################
    # Internal helpers #
    ####################

    def _configure_algorithm(self) -> None:
        """Configure the algorithm with scorer, rewriter, and callbacks."""
        self.algorithm.scorer = Scorer(
            model_callback=self.model_callback,
            metrics=self.metrics,
            max_concurrent=self.async_config.max_concurrent,
            optimizer_model=self.optimizer_model,
            throttle_seconds=float(self.async_config.throttle_value),
        )

        # Attach rewriter for mutation behavior
        # GEPA uses internal constant; other algorithms use MIPROV2 constant
        if isinstance(self.algorithm, GEPA):
            max_chars = GEPA_REWRITE_INSTRUCTION_MAX_CHARS
        else:
            self.algorithm.optimizer_model = self.optimizer_model
            max_chars = MIPROV2_REWRITE_INSTRUCTION_MAX_CHARS
        self.algorithm._rewriter = Rewriter(
            optimizer_model=self.optimizer_model,
            max_chars=max_chars,
            random_state=self.algorithm.random_state,
        )

        # Set status callback
        self.algorithm.status_callback = self._on_status
        # Set sub-step callback (updates the bottom progress row)
        self.algorithm.step_callback = self._on_step

    def _print_summary_table(self) -> None:
        console = Console(file=sys.stderr)

        if hasattr(self.algorithm, "generate_summary_table"):
            renderables = self.algorithm.generate_summary_table(
                self.optimization_report
            )
            console.print()
            for renderable in renderables:
                console.print(renderable)
            console.print()
        else:
            console.print(
                f"[dim]Optimization complete. (No summary table provided by {self.algorithm.name})[/]"
            )

    @contextmanager
    def _progress_context(self):
        """Context manager that sets up progress indicator if enabled."""
        if not self.display_config.show_indicator:
            yield
            return

        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            iter_task = progress.add_task(
                f"[bold white]Optimizing prompt with {self.algorithm.name}[/]"
            )
            step_task = progress.add_task("[rgb(55,65,81)]waiting...[/]")
            self._progress_state = (progress, iter_task, step_task)
            try:
                yield
            finally:
                self._progress_state = None

    def _handle_optimization_error(self, exc: Exception) -> None:
        total_steps: Optional[int] = None
        iterations: Optional[int] = getattr(self.algorithm, "iterations", None)
        if iterations is not None:
            total_steps = int(iterations)

        prefix = f"(iterations={iterations}) " if iterations is not None else ""
        detail = (
            f"{prefix}• error {exc.__class__.__name__}: {exc} "
            "• halted before first iteration"
        )

        self._on_status(
            RunnerStatusType.ERROR,
            detail=detail,
            step_index=None,
            total_steps=total_steps,
        )

        algo = self.algorithm.name
        raise DeepEvalError(f"[{algo}] {detail}") from None

    def _on_status(
        self,
        kind: RunnerStatusType,
        detail: str,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        """
        Unified status callback used by the algorithm.

        - PROGRESS: update the progress bar description and position
        - TIE:      optionally print a tie message
        - ERROR:    print a concise error message and allow the run to halt
        """
        algo = self.algorithm.name

        if kind is RunnerStatusType.ERROR:
            if self._progress_state is not None:
                progress, iter_task, step_task = self._progress_state
                if total_steps is not None:
                    progress.update(iter_task, total=total_steps)
                progress.update(
                    iter_task,
                    description=self._format_iter_description(
                        step_index, total_steps
                    ),
                )
                progress.update(
                    step_task, description=f"[rgb(255,85,85)]✕ {detail}[/]"
                )
            print(f"[{algo}] {detail}")
            return

        if kind is RunnerStatusType.TIE:
            if not self.display_config.announce_ties:
                return
            print(f"[{algo}] {detail}")
            return

        if kind is not RunnerStatusType.PROGRESS:
            return

        if self._progress_state is None:
            return

        progress, iter_task, step_task = self._progress_state

        if total_steps is not None:
            progress.update(iter_task, total=total_steps)

        if step_index is not None and step_index > 0:
            progress.advance(iter_task, 1)

        progress.update(
            iter_task,
            description=self._format_iter_description(step_index, total_steps),
        )

    def _on_step(self, label: str) -> None:
        if self._progress_state is None:
            return
        progress, _, step_task = self._progress_state
        progress.update(
            step_task, description=self._format_step_description(label)
        )

    def _format_iter_description(
        self,
        step_index: Optional[int],
        total_steps: Optional[int],
    ) -> str:
        algo = self.algorithm.name
        base = f"[bold white]Optimizing prompt with {algo}[/]"
        if step_index is not None and total_steps is not None:
            pct = int(100 * step_index / total_steps) if total_steps else 0
            return f"{base} [rgb(55,65,81)]iteration {step_index}/{total_steps} ({pct}%)[/]"
        return base

    def _format_step_description(self, label: str) -> str:
        if label:
            return f"[rgb(25,227,160)]⤷ {label}[/]"
        return ""
