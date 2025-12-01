from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.errors import DeepEvalError
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.evaluate.configs import AsyncConfig
from deepeval.optimization.adapters.deepeval_scoring_adapter import (
    DeepEvalScoringAdapter,
)
from deepeval.optimization.mutations.prompt_rewriter import (
    PromptRewriter,
)
from deepeval.optimization.types import (
    OptimizationReport,
    RunnerProtocol,
    RunnerStatusType,
)
from deepeval.optimization.utils import (
    validate_callback,
    validate_metrics,
    validate_instance,
    validate_sequence_of,
)
from deepeval.optimization.configs import (
    OptimizerDisplayConfig,
    PromptListMutationConfig,
)
from deepeval.prompt.prompt import Prompt
from deepeval.utils import get_or_create_event_loop
from deepeval.optimization.gepa.configs import GEPAConfig
from deepeval.optimization.gepa.loop import GEPARunner
from deepeval.optimization.miprov2.configs import MIPROConfig
from deepeval.optimization.miprov2.loop import MIPRORunner
from deepeval.optimization.copro.configs import COPROConfig
from deepeval.optimization.copro.loop import COPRORunner
from deepeval.optimization.simba.configs import SIMBAConfig
from deepeval.optimization.simba.loop import SIMBARunner


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
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
        async_config: Optional[AsyncConfig] = None,
        display_config: Optional[OptimizerDisplayConfig] = None,
        prompt_list_mutation_config: Optional[PromptListMutationConfig] = None,
        list_input_role: str = "user",
        algorithm: str = "gepa",
    ):
        # Validate and store the callback
        self.model_callback = validate_callback(
            component="PromptOptimizer",
            model_callback=model_callback,
        )
        self.metrics = validate_metrics(
            component="PromptOptimizer", metrics=metrics
        )
        # Validate async_config
        async_config = async_config or AsyncConfig()
        validate_instance(
            component="PromptOptimizer.__init__",
            param_name="async_config",
            value=async_config,
            expected_types=AsyncConfig,
        )
        self.async_config = async_config

        # validate display_config
        display_config = display_config or OptimizerDisplayConfig()
        validate_instance(
            component="PromptOptimizer.__init__",
            param_name="display_config",
            value=display_config,
            expected_types=OptimizerDisplayConfig,
        )
        self.display_config = display_config

        # validate prompt_list_mutation_config
        prompt_list_mutation_config = (
            prompt_list_mutation_config or PromptListMutationConfig()
        )
        validate_instance(
            component="PromptOptimizer.__init__",
            param_name="prompt_list_mutation_config",
            value=prompt_list_mutation_config,
            expected_types=PromptListMutationConfig,
        )
        self.prompt_list_mutation_config = prompt_list_mutation_config

        # validate list_input_role
        validate_instance(
            component="PromptOptimizer.__init__",
            param_name="list_input_role",
            value=list_input_role,
            expected_types=str,
        )
        self.list_input_role = list_input_role

        # Validate algorithm
        algo_raw = algorithm or "gepa"
        if not isinstance(algo_raw, str):
            raise DeepEvalError(
                "PromptOptimizer.__init__ expected `algorithm` to be a string "
                f"(e.g. 'gepa'), but received {type(algorithm).__name__!r} instead."
            )

        algo_normalized = (algo_raw.strip() or "gepa").lower()
        if algo_normalized in {"mipro", "miprov2"}:
            algo_normalized = "miprov2"

        self._allowed_algorithms = {"gepa", "miprov2", "copro", "simba"}

        if algo_normalized not in self._allowed_algorithms:
            raise DeepEvalError(
                "PromptOptimizer.__init__ received unsupported `algorithm` "
                f"value {algorithm!r}. Supported algorithms are: "
                + ", ".join(sorted(self._allowed_algorithms))
            )

        self.algorithm = algo_normalized

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
        # Validate prompt
        validate_instance(
            component="PromptOptimizer.optimize",
            param_name="prompt",
            value=prompt,
            expected_types=Prompt,
        )

        # Validate goldens: must be a list of Golden or ConversationalGolden
        validate_sequence_of(
            component="PromptOptimizer.optimize",
            param_name="goldens",
            value=goldens,
            expected_item_types=(Golden, ConversationalGolden),
            sequence_types=(list,),
        )

        if self.runner is None:
            self.set_runner(self._build_default_runner())

        if not self.display_config.show_indicator:
            best_prompt, report_dict = (
                self._run_optimization_with_error_handling(
                    prompt=prompt,
                    goldens=goldens,
                )
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

                try:
                    best_prompt, report_dict = (
                        self._run_optimization_with_error_handling(
                            prompt=prompt,
                            goldens=goldens,
                        )
                    )
                finally:
                    # Clear progress state even if an error occurs
                    self._progress_state = None

        best_prompt.optimization_report = OptimizationReport.from_runtime(
            report_dict
        )
        return best_prompt

    def set_runner(self, runner: RunnerProtocol):
        self._set_runner_callbacks(runner)
        scoring_adapter = getattr(runner, "scoring_adapter", None)
        if scoring_adapter is None:
            runner.scoring_adapter = self._build_default_scoring_adapter()
        else:
            if not len(runner.scoring_adapter.metrics):
                runner.scoring_adapter.set_metrics(self.metrics)
            if runner.scoring_adapter.model_callback is None:
                runner.scoring_adapter.model_callback = self.model_callback
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

    def _run_optimization_with_error_handling(
        self,
        *,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        """
        Run optimization and convert uncaught exceptions into a concise
        user facing error message.

        This is a fallback for errors that occur before the runner
        enters its main iteration loop, which would otherwise surface
        as a full traceback.
        """
        try:
            return self._run_optimization(prompt=prompt, goldens=goldens)
        except Exception as exc:
            # Try to recover iteration count from the runner config
            total_steps: Optional[int] = None
            iterations: Optional[int] = None
            runner_config = getattr(self.runner, "config", None)
            if runner_config is not None:
                iterations = getattr(runner_config, "iterations", None)
                if iterations is not None:
                    total_steps = int(iterations)

            prefix = (
                f"(iterations={iterations}) " if iterations is not None else ""
            )
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

            algo = self.algorithm.upper()

            # using `from None` avoids a long chained stack trace while keeping
            # the error message readable.
            raise DeepEvalError(f"[{algo}] {detail}") from None

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
        scoring_adapter = DeepEvalScoringAdapter(
            list_input_role=self.list_input_role
        )
        scoring_adapter.set_model_callback(self.model_callback)
        scoring_adapter.set_metrics(self.metrics)
        return scoring_adapter

    def _set_runner_callbacks(self, runner: RunnerProtocol):
        runner.model_callback = (
            self.model_callback
            if runner.model_callback is None
            else runner.model_callback
        )
        runner.status_callback = (
            self._on_status
            if runner.status_callback is None
            else runner.status_callback
        )

    def _build_default_runner(self) -> RunnerProtocol:
        if self.algorithm not in self._allowed_algorithms:
            raise DeepEvalError(
                f"Unsupported optimization algorithm: {self.algorithm!r}. "
                "Supported algorithms are: 'gepa', 'miprov2' (alias 'mipro'), "
                "'copro', 'simba'."
            )

        scoring_adapter = self._build_default_scoring_adapter()

        if hasattr(scoring_adapter, "configure_async"):
            scoring_adapter.configure_async(
                max_concurrent=self.async_config.max_concurrent,
                throttle_seconds=float(self.async_config.throttle_value),
            )

        if self.algorithm == "gepa":
            config = GEPAConfig()
            runner: RunnerProtocol = GEPARunner(
                config=config,
                scoring_adapter=scoring_adapter,
            )
        elif self.algorithm == "miprov2":
            # MIPROv2 0-shot, instruction-only
            config = MIPROConfig()
            runner = MIPRORunner(
                config=config,
                scoring_adapter=scoring_adapter,
            )
        elif self.algorithm == "copro":
            # COPRO cooperative multi-proposal variant
            config = COPROConfig()
            runner = COPRORunner(
                config=config,
                scoring_adapter=scoring_adapter,
            )
        else:
            config = SIMBAConfig()
            runner = SIMBARunner(
                config=config,
                scoring_adapter=scoring_adapter,
            )

        # Attach a PromptRewriter to the runner so that it has mutation behavior
        runner._rewriter = PromptRewriter(
            max_chars=config.rewrite_instruction_max_chars,
            list_mutation_config=self.prompt_list_mutation_config,
            random_state=runner.random_state,
        )

        self._set_runner_callbacks(runner)

        return runner
