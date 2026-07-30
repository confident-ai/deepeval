"""Telemetry smoke test.

Exercises every entrypoint that emits an `Evaluation` event and sends the
result to the real PostHog project, printing each payload as it goes.

    python a.py

See secrets/telemetry.md for what each property means.

No API keys and no LLM calls: the metric below is deterministic. It still goes
through `metric_progress_indicator`, which is the exact call path every
built-in metric uses to record telemetry, so the counters are real.
"""

import json
import time
from typing import Any, Dict, List

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from deepeval.tracing import observe, update_current_trace
from deepeval.tracing.integrations import Integration

import deepeval.telemetry as telemetry
from deepeval.telemetry import (
    LoginPromptSurface,
    capture_cli_command,
    capture_login_prompt_shown,
    capture_tracing_integration,
)


# --------------------------------------------------------------------------
# Print every event on its way to PostHog.
# --------------------------------------------------------------------------


class TeeBackend:
    """Forwards to the real backend and echoes the payload to stdout."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.count = 0

    def capture(self, anonymous_id: str, event: Any, properties: Dict) -> None:
        self.count += 1
        interesting = {
            key: value
            for key, value in sorted(properties.items())
            if key.startswith(("eval.", "judge.", "tracing.", "cli.", "login."))
            and value not in ([], 0, "none")
            or key == "eval.turn_kind"
        }
        print(f"\n  [{self.count}] {event.value}")
        for key, value in interesting.items():
            print(f"        {key:28} {json.dumps(value)}")
        self.inner.capture(anonymous_id, event, properties)

    def identify(self, anonymous_id: str, user_id: str) -> None:
        self.inner.identify(anonymous_id, user_id)

    def flush(self) -> None:
        self.inner.flush()


# --------------------------------------------------------------------------
# A metric with no LLM behind it.
# --------------------------------------------------------------------------


class _SmokeBody:
    """Shared body. Deliberately not a metric base class itself: `evaluate()`
    routes on isinstance, so inheriting both bases would send the multi-turn
    stub down the single-turn path."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.evaluation_model = "none (deterministic stub)"
        self.error = None
        self.score = None
        self.reason = None
        self.success = False
        self.evaluation_cost = 0.0
        self.skipped = False

    def measure(self, test_case, _show_indicator: bool = True, **kwargs):
        # The `with` block is what records the metric run. Every built-in
        # metric wraps `measure` this way.
        with metric_progress_indicator(self, _show_indicator=False):
            self.score = 1.0
            self.reason = "deterministic"
            self.success = True
            return self.score

    async def a_measure(self, test_case, _show_indicator: bool = True, **kw):
        return self.measure(test_case, _show_indicator=_show_indicator)

    def is_successful(self) -> bool:
        return bool(self.success)


class SmokeMetric(_SmokeBody, BaseMetric):
    @property
    def __name__(self):
        return "SmokeMetric"


class SmokeConversationalMetric(_SmokeBody, BaseConversationalMetric):
    @property
    def __name__(self):
        return "SmokeConversationalMetric"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
# 1. evaluate() -- eval.entrypoint = evaluate
# --------------------------------------------------------------------------


def scenario_evaluate() -> None:
    banner(
        "1. evaluate()  ->  expect entrypoint=evaluate, tracing.traced=false"
    )

    test_cases = [
        LLMTestCase(input=f"question {i}", actual_output=f"answer {i}")
        for i in range(3)
    ]
    evaluate(
        test_cases=test_cases,
        metrics=[SmokeMetric(), SmokeMetric()],
        display_config=_quiet_display(),
    )


# --------------------------------------------------------------------------
# 1b. evaluate() over conversations -- eval.turn_kind = multi_turn
# --------------------------------------------------------------------------


def scenario_evaluate_multi_turn() -> None:
    banner("1b. evaluate() multi-turn  ->  expect turn_kind=multi_turn")

    test_cases = [
        ConversationalTestCase(
            turns=[
                Turn(role="user", content=f"hello {i}"),
                Turn(role="assistant", content=f"hi {i}"),
            ]
        )
        for i in range(2)
    ]
    evaluate(
        test_cases=test_cases,
        metrics=[SmokeConversationalMetric()],
        display_config=_quiet_display(),
    )


# --------------------------------------------------------------------------
# 2. evals_iterator() -- eval.entrypoint = evals_iterator, WITH tracing
# --------------------------------------------------------------------------


@observe()
def traced_app(question: str) -> str:
    """An @observe span, so the iterator run actually produces traces."""
    answer = f"answer to {question}"
    update_current_trace(input=question, output=answer)
    return answer


def scenario_evals_iterator() -> None:
    banner(
        "2. evals_iterator()  ->  expect entrypoint=evals_iterator, "
        "tracing.traced=true"
    )

    dataset = EvaluationDataset(
        goldens=[Golden(input=f"traced question {i}") for i in range(3)]
    )
    for golden in dataset.evals_iterator(metrics=[SmokeMetric()]):
        traced_app(golden.input)


# --------------------------------------------------------------------------
# 3. Bare metric.measure() -- eval.entrypoint = standalone
# --------------------------------------------------------------------------


def scenario_standalone_metrics() -> None:
    banner(
        "3. bare metric.measure() x4  ->  expect ONE event at flush, "
        "entrypoint=standalone"
    )

    metric = SmokeMetric()
    for i in range(4):
        metric.measure(LLMTestCase(input=f"bare {i}", actual_output=f"out {i}"))
    print("        (nothing sent yet -- these are buffered in memory)")
    telemetry.flush_standalone_metrics()


# --------------------------------------------------------------------------
# 4. The non-evaluation events
# --------------------------------------------------------------------------


def scenario_other_events() -> None:
    banner("4. CLI command, integration install, login prompt")

    capture_cli_command("view", {"view": None})
    # Fires once per process no matter how many handlers get constructed.
    for _ in range(3):
        with capture_tracing_integration(Integration.LANGCHAIN):
            pass
    capture_login_prompt_shown(LoginPromptSurface.POST_EVAL)


def _quiet_display():
    from deepeval.evaluate.configs import DisplayConfig

    return DisplayConfig(show_indicator=False, print_results=False)


def main() -> None:
    if telemetry.telemetry_opt_out():
        raise SystemExit(
            "DEEPEVAL_TELEMETRY_OPT_OUT is set -- unset it or nothing is sent."
        )

    real = telemetry.get_backend()
    if isinstance(real, telemetry.NoopBackend):
        raise SystemExit(
            "Telemetry backend is a no-op; the posthog client failed to build."
        )

    tee = TeeBackend(real)
    telemetry.set_backend(tee)

    anonymous_id = telemetry.get_unique_id()
    print(f"\nSending as user.unique_id = {anonymous_id}")
    print("Filter the PostHog dashboard on that id to see only this run.")

    scenario_evaluate()
    scenario_evaluate_multi_turn()
    scenario_evals_iterator()
    scenario_standalone_metrics()
    scenario_other_events()

    banner(f"Flushing {tee.count} events to PostHog")
    telemetry.flush()
    # The client ships on a background thread; give it a moment to drain.
    time.sleep(3)
    print(
        f"\nDone. {tee.count} events sent as {anonymous_id}.\n"
        "PostHog ingestion usually lags a few seconds to a minute.\n"
    )


if __name__ == "__main__":
    main()
