"""Telemetry smoke test.

Exercises every entrypoint that emits an `Evaluation` event, printing each
payload and writing it to `telemetry.jsonl` for inspection.

    python a.py              # local only, nothing leaves the machine
    python a.py --posthog    # also send to the real PostHog project

See secrets/telemetry.md for what each property means.

No API keys and no LLM calls: the metric below is deterministic. It still goes
through `metric_progress_indicator`, which is the exact call path every
built-in metric uses to record telemetry, so the counters are real.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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


LOG_PATH = Path(__file__).with_name("telemetry.jsonl")

# Shown inline. Everything else still goes to the file.
_INTERESTING_PREFIXES = ("eval.", "judge.", "tracing.", "cli.", "login.")


class RecordingBackend:
    """Writes every event to a file, and optionally forwards it upstream.

    Implements the same three methods as `PostHogBackend`, which is all
    `TelemetryBackend` requires -- the point of the protocol is that the rest
    of the library cannot tell the difference.
    """

    def __init__(self, path: Path, forward_to: Optional[Any] = None) -> None:
        self.path = path
        self.forward_to = forward_to
        self.events: List[Dict[str, Any]] = []
        self.handle = path.open("w")

    def capture(self, anonymous_id: str, event: Any, properties: Dict) -> None:
        # The standalone accumulator's atexit hook can fire after close().
        if self.handle.closed:
            return
        record = {
            "seq": len(self.events) + 1,
            "at": time.strftime("%H:%M:%S"),
            "event": event.value,
            "anonymous_id": anonymous_id,
            "properties": properties,
        }
        self.events.append(record)
        self.handle.write(json.dumps(record, default=str) + "\n")
        self.handle.flush()

        shown = {
            key: value
            for key, value in sorted(properties.items())
            if (
                key.startswith(_INTERESTING_PREFIXES)
                and value not in ([], 0, "none")
            )
            or key == "eval.turn_kind"
        }
        print(f"\n  [{record['seq']}] {event.value}")
        for key, value in shown.items():
            print(f"        {key:28} {json.dumps(value)}")

        if self.forward_to is not None:
            self.forward_to.capture(anonymous_id, event, properties)

    def flush(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
        if self.forward_to is not None:
            self.forward_to.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    @property
    def count(self) -> int:
        return len(self.events)

    def summary(self) -> str:
        by_event = Counter(record["event"] for record in self.events)
        runs = {
            record["properties"]["eval.run_id"]
            for record in self.events
            if "eval.run_id" in record["properties"]
        }
        lines = [f"  {count:>2}x {name}" for name, count in by_event.items()]
        lines.append(f"  {len(runs):>2} distinct eval.run_id")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# A metric with no LLM behind it.
# --------------------------------------------------------------------------


class _SmokeBody:
    """Shared body. Deliberately not a metric base class itself: `evaluate()`
    routes on isinstance, so inheriting both bases would send the multi-turn
    stub down the single-turn path."""

    def __init__(self, threshold: float = 0.5, model: Any = None):
        # Must be set here, not assigned afterwards: evaluate() rebuilds the
        # metric per test case, so a later assignment never reaches the
        # indicator that records the judge.
        self.model = model
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
# 0. Install integrations first, so the runs below carry them
# --------------------------------------------------------------------------


def scenario_install_integrations() -> None:
    """Must run before any evaluation.

    `tracing.integrations` is a snapshot of what is registered at the moment
    the run finishes, so installing an integration afterwards leaves every
    Evaluation event reporting no integration at all.
    """
    banner("0. install integrations  ->  every later run carries them")

    for integration in (Integration.LANGCHAIN, Integration.OPEN_AI):
        # Repeats are deduplicated, so this fires once per integration.
        for _ in range(3):
            with capture_tracing_integration(integration):
                pass


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


def scenario_judge_model() -> None:
    """judge.provider / judge.model only fill in when a metric carries a model.

    Nothing is sent to the provider: the stub never calls the model, it only
    needs the object so the judge dimension can be derived from its class.
    """
    banner(
        "4. metric with a judge model  ->  expect judge.provider/judge.model"
    )

    # Constructing the client needs a key present; nothing is ever sent to it.
    os.environ.setdefault("OPENAI_API_KEY", "sk-unused-by-this-script")
    try:
        from deepeval.models import OpenAIModel

        model = OpenAIModel(model="gpt-4.1")
    except Exception as error:
        print(f"        skipped, could not construct a model: {error}")
        return

    evaluate(
        test_cases=[LLMTestCase(input="judged", actual_output="output")],
        metrics=[SmokeMetric(model=model)],
        display_config=_quiet_display(),
    )


def scenario_remaining_entrypoints() -> None:
    """The two entrypoints this process cannot reach naturally.

    `compare` needs an arena test case and a real arena metric, and `pytest`
    only exists inside a `deepeval test run` session. Both scopes are opened
    directly here so every entrypoint appears on the dashboard.
    """
    banner("4b. compare + pytest scopes  ->  the remaining two entrypoints")

    for entrypoint in (
        telemetry.Entrypoint.COMPARE,
        telemetry.Entrypoint.PYTEST,
    ):
        with telemetry.capture_evaluation_run(entrypoint):
            for index in range(2):
                telemetry.record_test_case(
                    LLMTestCase(input=f"case {index}", actual_output="out")
                )
                telemetry.record_metric(
                    "SmokeMetric", async_mode=False, in_component=False
                )


def scenario_other_events() -> None:
    banner("5. CLI command, login, other features")

    capture_cli_command("view", {"view": None})
    capture_login_prompt_shown(LoginPromptSurface.POST_EVAL)

    # An abandoned login is a distinct outcome from a completed one.
    with telemetry.capture_login_event() as span:
        span.set_method(telemetry.LoginMethod.BROWSER)
        span.set_outcome(telemetry.LoginOutcome.COMPLETED)

    with telemetry.capture_synthesizer_run(
        method="docs", max_generations=10, num_evolutions=2, evolutions={}
    ):
        pass
    with telemetry.capture_conversation_simulator_run(num_conversations=3):
        pass
    with telemetry.capture_benchmark_run(benchmark="MMLU", num_tasks=5):
        pass


def _quiet_display():
    from deepeval.evaluate.configs import DisplayConfig

    return DisplayConfig(show_indicator=False, print_results=False)


def main() -> None:
    to_posthog = "--posthog" in sys.argv

    if telemetry.telemetry_opt_out():
        raise SystemExit(
            "DEEPEVAL_TELEMETRY_OPT_OUT is set -- unset it or nothing is "
            "captured at all."
        )

    upstream = None
    if to_posthog:
        upstream = telemetry.get_backend()
        if isinstance(upstream, telemetry.NoopBackend):
            raise SystemExit(
                "Telemetry backend is a no-op; the posthog client failed to "
                "build."
            )

    recorder = RecordingBackend(LOG_PATH, forward_to=upstream)
    telemetry.set_backend(recorder)

    anonymous_id = telemetry.get_unique_id()
    print(f"\nuser.unique_id = {anonymous_id}")
    print(f"Writing to {LOG_PATH}")
    if to_posthog:
        print("Also sending to PostHog. Filter the dashboard on that id.")
    else:
        print(
            "Local only -- nothing leaves this machine. Use --posthog to send."
        )

    scenario_install_integrations()
    scenario_evaluate()
    scenario_evaluate_multi_turn()
    scenario_evals_iterator()
    scenario_standalone_metrics()
    scenario_judge_model()
    scenario_remaining_entrypoints()
    scenario_other_events()

    # The standalone accumulator also flushes at process exit, but doing it
    # here keeps the file complete before the summary is printed.
    telemetry.flush_standalone_metrics()
    telemetry.flush()
    if to_posthog:
        # The posthog client ships on a background thread.
        time.sleep(3)

    banner(f"{recorder.count} events")
    print(recorder.summary())
    recorder.close()
    print(
        f"\nFull payloads: {LOG_PATH}\n"
        f"  cat {LOG_PATH.name} | jq .\n"
        f"  cat {LOG_PATH.name} | jq -r '.properties.\"eval.entrypoint\"'\n"
    )


if __name__ == "__main__":
    main()
