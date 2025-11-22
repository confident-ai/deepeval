from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from deepeval.errors import DeepEvalError
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.evaluate.configs import AsyncConfig
from deepeval.optimization.adapters.deepeval_scoring_adapter import (
    DeepEvalScoringAdapter,
)
from deepeval.optimization.gepa.configs import GEPAConfig
from deepeval.optimization.gepa.loop import GEPARunner
from deepeval.optimization.types import (
    OptimizationReport,
    RunnerProtocol,
    RunnerStatusType,
)
from deepeval.optimization.utils import validate_callback
from deepeval.prompt.prompt import Prompt
from deepeval.utils import get_or_create_event_loop
from .configs import OptimizerDisplayConfig


if TYPE_CHECKING:
    from deepeval.dataset.golden import Golden, ConversationalGolden


class PromptOptimizer:
    """
    High-level entrypoint for prompt optimization.

    Typical usage:

        optimizer = PromptOptimizer(
            metrics=[AnswerRelevancyMetric()],
            model_callback=model_callback,
        )

        optimized_prompt = optimizer.optimize(
            prompt=Prompt(text_template="Respond to the query."),
            goldens=goldens,
        )

    By default, this constructs and uses a GEPA based runner internally.
    Advanced users can construct their own runner with a custom config
    (GEPAConfig) and attach it via `set_runner(...)`.
    """

    def __init__(
        self,
        *,
        model_callback: Callable[
            ...,
            Union[
                str,
                Dict,
                Tuple[Union[str, Dict], float],
            ],
        ],
        # used if scoring_adapter is None
        metrics: Optional[
            Union[List[BaseMetric], List[BaseConversationalMetric]]
        ] = None,
        async_config: Optional[AsyncConfig] = None,
        display_config: Optional[OptimizerDisplayConfig] = None,
        algorithm: str = "gepa",
    ):
        # Validate and store the callback
        self.model_callback = validate_callback(
            component="PromptOptimizer",
            model_callback=model_callback,
        )

        # Metrics are required for the default GEPA runner
        # They are marked optional on the API for future custom runner support.
        self.metrics: Optional[
            Union[List[BaseMetric], List[BaseConversationalMetric]]
        ] = (list(metrics) if metrics is not None else None)

        self.async_config = async_config or AsyncConfig()
        self.display_config = display_config or OptimizerDisplayConfig()
        self.algorithm = (algorithm or "gepa").lower()

        # Internal state used only when a progress indicator is active.
        # Tuple is (Progress instance, task_id).
        self._progress_state: Optional[Tuple[Progress, int]] = None

        self.runner: Optional[RunnerProtocol] = None

    ##############
    # Public API #
    ##############

    def optimize(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Prompt:
        """
        Run the configured optimization algorithm and return an optimized Prompt.

        The returned Prompt will have an OptimizationReport attached as
        `prompt.optimization_report`.
        """
        self.runner = self.runner or self._build_default_runner()

        if not self.display_config.show_indicator:
            best_prompt, report_dict = self._run_optimization(
                prompt=prompt, goldens=goldens
            )
        else:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                # Total will be provided by the runner via the
                # progress status_callback. Start at 0 and update later.
                task = progress.add_task(
                    f"Optimizing prompt with {self.algorithm.upper()}..."
                )
                self._progress_state = (progress, task)

                best_prompt, report_dict = self._run_optimization(
                    prompt=prompt, goldens=goldens
                )

            self._progress_state = None

        best_prompt.optimization_report = OptimizationReport.from_runtime(
            report_dict
        )
        return best_prompt

    def set_runner(self, runner: RunnerProtocol):
        self._set_runner_callbacks(runner)
        self.runner = runner

    ####################
    # Internal helpers #
    ####################

    def _run_optimization(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        if self.async_config.run_async:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.runner.a_execute(prompt=prompt, goldens=goldens)
            )
        return self.runner.execute(prompt=prompt, goldens=goldens)

    def _on_status(
        self,
        kind: RunnerStatusType,
        *,
        detail: str,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        """
        Unified status callback used by the configured runner.

        - PROGRESS: update the progress bar description and position
        - TIE:      optionally print a tie message
        - ERROR:    print a concise error message and allow the run to halt
        """
        algo = self.algorithm.upper()

        # ERROR: always print, optionally update progress bar
        if kind is RunnerStatusType.ERROR:
            if (
                self.display_config.show_indicator
                and self._progress_state is not None
            ):
                progress, task = self._progress_state

                if total_steps is not None:
                    progress.update(task, total=total_steps)

                description = self._format_progress_description(detail)
                progress.update(task, description=description)

            # Print a concise, error line regardless of indicator state
            print(f"[{algo}] {detail}")
            return

        # TIE: optional one line message, no progress bar changes
        if kind is RunnerStatusType.TIE:
            if not self.display_config.announce_ties:
                return
            print(f"[{algo}] {detail}")
            return

        if kind is not RunnerStatusType.PROGRESS:
            return

        if not self.display_config.show_indicator:
            return

        if self._progress_state is None:
            return

        progress, task = self._progress_state

        # Allow the runner to set or update the total steps.
        if total_steps is not None:
            progress.update(task, total=total_steps)

        # iteration 0 shouldn't advance the bar
        if step_index is not None and step_index > 0:
            progress.advance(task, 1)

        description = self._format_progress_description(detail)
        progress.update(task, description=description)

    def _format_progress_description(self, detail: str) -> str:
        """
        Compose a human readable progress line using an algorithm agnostic
        prefix and an algorithm specific detail string provided by the runner.
        """
        algo = self.algorithm.upper()
        base = f"Optimizing prompt with {algo}"
        if detail:
            return f"{base} [rgb(25,227,160)]{detail}[/]"
        return base

    def _build_default_scoring_adapter(self) -> DeepEvalScoringAdapter:
        if not self.metrics:
            raise DeepEvalError(
                "PromptOptimizer requires `metrics` when using the default "
                "GEPA algorithm. Pass a list of metrics when constructing "
                "the optimizer."
            )
        return DeepEvalScoringAdapter(
            model_callback=self.model_callback,
            metrics=self.metrics,
        )

    def _set_runner_callbacks(self, runner: RunnerProtocol):
        runner.model_callback = self.model_callback
        runner.status_callback = self._on_status

    def _build_default_runner(self) -> RunnerProtocol:
        if self.algorithm != "gepa":
            raise DeepEvalError(
                f"Unsupported optimization algorithm: {self.algorithm!r}. "
                "Only 'gepa' is currently supported."
            )

        scoring_adapter = self._build_default_scoring_adapter()

        if hasattr(scoring_adapter, "configure_async"):
            scoring_adapter.configure_async(
                max_concurrent=self.async_config.max_concurrent,
                throttle_seconds=float(self.async_config.throttle_value),
            )

        config = GEPAConfig()
        runner = GEPARunner(config=config, scoring_adapter=scoring_adapter)

        self._set_runner_callbacks(runner)

        return runner
