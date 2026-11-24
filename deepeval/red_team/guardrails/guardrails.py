from typing import List, Any, Optional
import time
import asyncio
from typing import Literal
from pydantic import BaseModel

from deepteam.guardrails.guards.base_guard import BaseGuard
from deepteam.telemetry import capture_guardrail_run


class GuardVerdict(BaseModel):
    name: str
    safety_level: Literal["safe", "borderline", "unsafe", "uncertain"]
    latency: Optional[float] = None
    reason: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None


class GuardResult:
    def __init__(self, verdicts: List[GuardVerdict]):
        # Per-guard verdicts and data
        self.verdicts: List[GuardVerdict] = verdicts
        # Breach if any guard verdict indicates not explicitly safe
        self.breached = self._compute_breached()

    def _normalize_level(self, level: Any) -> Optional[str]:
        if level is None:
            return None
        value = getattr(level, "value", level)
        if isinstance(value, str):
            return value.lower()
        return None

    def _compute_breached(self) -> bool:
        for result in self.verdicts:
            level = self._normalize_level(result.safety_level)
            if level in ("unsafe", "borderline", "uncertain"):
                return True
        return False


class Guardrails:
    """
    Open-source guardrails system for production LLM safety.
    Fast binary classification to guard inputs and outputs.
    """

    def __init__(
        self,
        input_guards: List[BaseGuard],
        output_guards: List[BaseGuard],
        evaluation_model: str = "gpt-4.1",
        sample_rate: float = 1.0,
    ):
        """
        Initialize Guardrails with separate input and output guards.

        Args:
            input_guards: List of guards to check inputs before they reach your LLM
            output_guards: List of guards to check outputs before they reach your users
            evaluation_model: OpenAI model to use for guard evaluation (default: gpt-4.1)
            sample_rate: Fraction of requests to actually guard (0.0 to 1.0, default: 1.0)
        """
        # Validate sample_rate
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {sample_rate}"
            )

        self.sample_rate = sample_rate
        self.evaluation_model = evaluation_model
        self._request_count = 0

        # Update all guards to use the specified evaluation model
        self.input_guards = self._update_guards_model(
            input_guards, evaluation_model
        )
        self.output_guards = self._update_guards_model(
            output_guards, evaluation_model
        )

    def _update_guards_model(
        self, guards: List[BaseGuard], evaluation_model: str
    ) -> List[BaseGuard]:
        """Update all guards to use the specified evaluation model"""
        from deepteam.guardrails.guards.base_guard import initialize_model

        for guard in guards:
            guard.model, guard.using_native_model = initialize_model(
                evaluation_model
            )
            guard.evaluation_model = guard.model.get_model_name()

        return guards

    def _should_process(self) -> bool:
        """Deterministic sampling: exactly sample_rate fraction of requests"""
        self._request_count += 1
        if self.sample_rate == 0.0:
            return False
        if self.sample_rate == 1.0:
            return True

        interval = int(1 / self.sample_rate)
        return self._request_count % interval == 0

    def guard_input(self, input: str) -> GuardResult:
        """
        Guard an input string using all configured input guards.
        Returns GuardResult with binary safe/unsafe classification.
        """
        if len(self.input_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard inputs when no input_guards are provided."
            )

        if not self._should_process():
            return GuardResult(verdicts=[])

        verdicts: List[GuardVerdict] = []

        with capture_guardrail_run(
            type="input",
            guards=[guard.__name__ for guard in self.input_guards],
        ):
            for guard in self.input_guards:
                start_time = time.time()
                try:
                    _ = guard.guard_input(input)
                    latency = time.time() - start_time

                    safety_level = guard.safety_level or "unsafe"
                    if hasattr(safety_level, "value"):
                        safety_level = safety_level.value

                    verdicts.append(
                        GuardVerdict(
                            name=guard.__name__,
                            safety_level=safety_level,  # type: ignore[arg-type]
                            latency=latency,
                            reason=guard.reason,
                            score=guard.score,
                        )
                    )

                except Exception as e:
                    verdicts.append(
                        GuardVerdict(
                            name=guard.__name__,
                            safety_level="unsafe",
                            latency=time.time() - start_time,
                            error=str(e),
                            score=0.0,
                        )
                    )

        return GuardResult(verdicts=verdicts)

    def guard_output(self, input: str, output: str) -> GuardResult:
        """
        Guard an output string using all configured output guards.
        Returns GuardResult with binary safe/unsafe classification.
        """
        if len(self.output_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard outputs when no output_guards are provided."
            )

        if not self._should_process():
            return GuardResult(verdicts=[])

        verdicts: List[GuardVerdict] = []

        with capture_guardrail_run(
            type="output",
            guards=[guard.__name__ for guard in self.output_guards],
        ):
            for guard in self.output_guards:
                start_time = time.time()
                try:
                    _ = guard.guard_output(input, output)
                    latency = time.time() - start_time

                    safety_level = guard.safety_level or "unsafe"
                    if hasattr(safety_level, "value"):
                        safety_level = safety_level.value

                    verdicts.append(
                        GuardVerdict(
                            name=guard.__name__,
                            safety_level=safety_level,  # type: ignore[arg-type]
                            latency=latency,
                            reason=guard.reason,
                            score=guard.score,
                        )
                    )

                except Exception as e:
                    verdicts.append(
                        GuardVerdict(
                            name=guard.__name__,
                            safety_level="unsafe",
                            latency=time.time() - start_time,
                            error=str(e),
                            score=0.0,
                        )
                    )

        return GuardResult(verdicts=verdicts)

    async def a_guard_input(self, input: str) -> GuardResult:
        """
        Async version of guard_input for better performance.
        """
        if len(self.input_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard inputs when no input_guards are provided."
            )

        if not self._should_process():
            return GuardResult(verdicts=[])

        tasks = []
        for guard in self.input_guards:
            task = asyncio.create_task(
                self._async_guard_input_single(guard, input)
            )
            tasks.append((guard, task))

        verdicts: List[GuardVerdict] = []

        for guard, task in tasks:
            try:
                result = await task
                verdicts.append(result)
            except Exception as e:
                verdicts.append(
                    GuardVerdict(
                        name=guard.__name__,
                        safety_level="unsafe",
                        error=str(e),
                        score=0.0,
                    )
                )

        return GuardResult(verdicts=verdicts)

    async def a_guard_output(self, input: str, output: str) -> GuardResult:
        """
        Async version of guard_output for better performance.
        """
        if len(self.output_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard outputs when no output_guards are provided."
            )

        if not self._should_process():
            return GuardResult(verdicts=[])

        tasks = []
        for guard in self.output_guards:
            task = asyncio.create_task(
                self._async_guard_output_single(guard, input, output)
            )
            tasks.append((guard, task))

        verdicts: List[GuardVerdict] = []

        for guard, task in tasks:
            try:
                result = await task
                verdicts.append(result)
            except Exception as e:
                verdicts.append(
                    GuardVerdict(
                        name=guard.__name__,
                        safety_level="unsafe",
                        error=str(e),
                        score=0.0,
                    )
                )

        return GuardResult(verdicts=verdicts)

    async def _async_guard_input_single(self, guard: BaseGuard, input: str):
        """Helper method for async input guarding."""
        start_time = time.time()
        _ = await guard.a_guard_input(input)
        latency = time.time() - start_time

        safety_level = guard.safety_level or "unsafe"
        if hasattr(safety_level, "value"):
            safety_level = safety_level.value

        return GuardVerdict(
            name=guard.__name__,
            safety_level=safety_level,  # type: ignore[arg-type]
            latency=latency,
            reason=guard.reason,
            score=guard.score,
        )

    async def _async_guard_output_single(
        self, guard: BaseGuard, input: str, output: str
    ):
        """Helper method for async output guarding."""
        start_time = time.time()
        _ = await guard.a_guard_output(input, output)
        latency = time.time() - start_time

        safety_level = guard.safety_level or "unsafe"
        if hasattr(safety_level, "value"):
            safety_level = safety_level.value

        return GuardVerdict(
            name=guard.__name__,
            safety_level=safety_level,  # type: ignore[arg-type]
            latency=latency,
            reason=guard.reason,
            score=guard.score,
        )
