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
from deepeval.metrics.utils import initialize_model
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.scorer import Scorer
from deepeval.optimizer.rewriter import Rewriter
from deepeval.optimizer.types import (
    OptimizationReport,
    RunnerStatusType,
)
from deepeval.optimizer.utils import (
    validate_callback,
    validate_metrics,
)
from deepeval.optimizer.configs import (
    DisplayConfig,
    MutationConfig,
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


class PromptOptimizer:
    """
    High-level entrypoint for prompt optimization.

    Typical usage:

        optimizer = PromptOptimizer(
            optimizer_model=GPTModel(),  # for internal prompt rewriting
            model_callback=my_callback,  # for running your AI app during scoring
            metrics=[AnswerRelevancyMetric()],
        )

        optimized_prompt = optimizer.optimize(
            prompt=Prompt(text_template="Respond to the query."),
            goldens=goldens,
        )

    The `optimizer_model` is used internally for prompt rewriting/mutation.
    The `model_callback` is called during scoring to run your AI app.
    Your callback receives `prompt` and `golden` - call `prompt.interpolate(...)`
    with the golden's input to get the final prompt to send to your model.

    By default uses GEPA algorithm. Pass a different algorithm instance
    (MIPROV2, COPRO, SIMBA) to use a different optimization strategy.
    """

    def __init__(
        self,
        model_callback: Callable[..., str],
        metrics: Union[List[BaseMetric], List[BaseConversationalMetric]],
        optimizer_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        algorithm: Union[GEPA, MIPROV2, COPRO, SIMBA] = GEPA(),
        async_config: Optional[AsyncConfig] = AsyncConfig(),
        display_config: Optional[DisplayConfig] = DisplayConfig(),
        mutation_config: Optional[MutationConfig] = MutationConfig(),
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
        self.mutation_config = mutation_config
        self.algorithm = algorithm
        self._configure_algorithm()

        # Internal state used only when a progress indicator is active.
        # Tuple is (Progress instance, task_id).
        self._progress_state: Optional[Tuple[Progress, int]] = None

    ##############
    # Public API #
    ##############

    def optimize(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Prompt:
        """
        Run the configured optimization algorithm and return an optimized Prompt.

        The returned Prompt will have an OptimizationReport attached as
        `prompt.optimization_report`.
        """
        # DEBUG: Log original prompt
        print(f"\n[DEBUG] Starting optimization with {self.algorithm.name}")
        print(
            f"[DEBUG] Original prompt: {prompt.text_template[:200] if prompt.text_template else prompt.messages_template}..."
        )
        print(f"[DEBUG] Number of goldens: {len(goldens)}")

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
                # Total will be provided by the algorithm via the
                # progress status_callback. Start at 0 and update later.
                task = progress.add_task(
                    f"Optimizing prompt with {self.algorithm.name}..."
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

        # DEBUG: Log optimization results
        accepted_count = len(report_dict.get("accepted_iterations", []))
        print(f"\n[DEBUG] Optimization complete!")
        print(
            f"[DEBUG] Accepted iterations (children that beat parent): {accepted_count}"
        )
        print(
            f"[DEBUG] Total prompt configurations explored: {len(report_dict.get('prompt_configurations', {}))}"
        )
        print(
            f"[DEBUG] Best prompt: {best_prompt.text_template[:200] if best_prompt.text_template else best_prompt.messages_template}..."
        )

        # Check if prompt changed
        original_text = prompt.text_template or str(prompt.messages_template)
        best_text = best_prompt.text_template or str(
            best_prompt.messages_template
        )
        if original_text.strip() == best_text.strip():
            print(
                f"[DEBUG] ⚠️  WARNING: Optimized prompt is IDENTICAL to original!"
            )
            print(
                f"[DEBUG] This can happen if: (1) no child prompts scored higher than parent, or (2) rewriter returned same prompt"
            )
        else:
            print(f"[DEBUG] ✓ Prompt was modified during optimization")

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
            throttle_seconds=float(self.async_config.throttle_value),
        )

        # Attach rewriter for mutation behavior
        self.algorithm._rewriter = Rewriter(
            optimizer_model=self.optimizer_model,
            max_chars=self.algorithm.config.rewrite_instruction_max_chars,
            list_mutation_config=self.mutation_config,
            random_state=self.algorithm.random_state,
        )

        # Set status callback
        self.algorithm.status_callback = self._on_status

    def _run_optimization(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        if self.async_config.run_async:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.algorithm.a_execute(prompt=prompt, goldens=goldens)
            )
        return self.algorithm.execute(prompt=prompt, goldens=goldens)

    def _run_optimization_with_error_handling(
        self,
        prompt: Prompt,
        goldens: Union[List["Golden"], List["ConversationalGolden"]],
    ) -> Tuple[Prompt, Dict]:
        """
        Run optimization and convert uncaught exceptions into a concise
        user facing error message.

        This is a fallback for errors that occur before the algorithm
        enters its main iteration loop, which would otherwise surface
        as a full traceback.
        """
        try:
            return self._run_optimization(prompt=prompt, goldens=goldens)
        except Exception as exc:
            # Try to recover iteration count from the algorithm config
            total_steps: Optional[int] = None
            iterations: Optional[int] = None
            if self.algorithm.config is not None:
                iterations = getattr(self.algorithm.config, "iterations", None)
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

            algo = self.algorithm.name

            # using `from None` avoids a long chained stack trace while keeping
            # the error message readable.
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

        # Allow the algorithm to set or update the total steps.
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
        prefix and an algorithm specific detail string provided by the algorithm.
        """
        algo = self.algorithm.name
        base = f"Optimizing prompt with {algo}"
        if detail:
            return f"{base} [rgb(25,227,160)]{detail}[/]"
        return base
