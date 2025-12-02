import io
import time
import asyncio
from types import SimpleNamespace
from typing import Callable, List, Optional, Protocol, runtime_checkable

from deepeval.constants import ProviderSlug as PS
from deepeval.metrics import BaseMetric, TaskCompletionMetric
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.optimization.types import ModuleId
from deepeval.prompt.prompt import Prompt
from deepeval.tracing.types import TraceSpanStatus


@runtime_checkable
class ApiTestCaseLike(Protocol):
    name: Optional[str]
    success: Optional[bool]
    metrics_data: List
    input: Optional[str]
    actual_output: Optional[str]
    expected_output: Optional[str]
    context: Optional[List[str]]
    retrieval_context: Optional[List[str]]

    def update_metric_data(self, *args, **kwargs) -> None: ...
    def update_status(self, *args, **kwargs) -> None: ...
    def update_run_duration(self, *args, **kwargs) -> None: ...


def make_trace_api_like(status):
    """Shape compatible with TraceApi members that `execute` touches."""
    return SimpleNamespace(
        name="trace",
        status=status,
        error=None,
        input=None,
        output=None,
        expected_output=None,
        context=None,
        retrieval_context=None,
        agent_spans=[],
        llm_spans=[],
        retriever_spans=[],
        tool_spans=[],
        base_spans=[],
        metrics_data=[],
    )


def make_span_api_like():
    return SimpleNamespace(status=None, error=None, metrics_data=[])


##########
# Models #
##########


class StubProvider:
    def __init__(self, value: str) -> None:
        self.value = value


class StubModelSettings:
    def __init__(self, provider=None, name: str | None = None) -> None:
        self.provider = provider
        self.name = name


class StubPrompt:
    def __init__(
        self,
        alias: str | None = None,
        label: str | None = None,
        model_settings: StubModelSettings | None = None,
    ) -> None:
        self.alias = alias
        self.label = label
        self.model_settings = model_settings


class DummyModel:
    pass


class AlwaysJsonModel:
    """
    Test stub that always returns JSON text and NEVER accepts `schema=`,
    so the simulator takes the JSON path (trimAndLoadJson).

    Pass an `extractor` callable that takes the full prompt and returns the
    JSON snippet to emit.

    Usage:
      - AlwaysJsonModel.balanced_json_after_anchor(anchor)

    """

    def __init__(self, extractor: Callable[[str], str]):
        if not callable(extractor):
            raise TypeError("extractor must be a callable(prompt) -> str JSON")
        self._extractor = extractor

    # no support for `schema=` kwarg so we always take JSON path
    def generate(self, prompt: str) -> str:
        return self._extractor(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "always-json-stub"

    @staticmethod
    def balanced_json_after_anchor(anchor_text: str) -> Callable[[str], str]:
        """
        Returns an extractor that finds the first balanced JSON object
        after the given anchor string.
        """

        def extractor(prompt: str) -> str:
            anchor_index = prompt.find(anchor_text)
            if anchor_index == -1:
                raise ValueError(f"Anchor '{anchor_text}' not found in prompt.")

            json_start_index = prompt.find("{", anchor_index)
            if json_start_index == -1:
                raise ValueError(
                    f"No opening '{{' found after anchor '{anchor_text}'."
                )

            brace_depth = 0
            for char_index, character in enumerate(
                prompt[json_start_index:], start=json_start_index
            ):
                if character == "{":
                    brace_depth += 1
                elif character == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        json_end_index = char_index + 1
                        return prompt[json_start_index:json_end_index]

            raise ValueError(f"Unbalanced braces after anchor '{anchor_text}'.")

        return extractor


class _RecordingClient:
    """
    Generic SDK-style client stub that records kwargs passed to its constructor.

    Used by provider model tests to assert that we pass the correct api_key and
    retry options to SDK constructors without making network calls.
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


###########
# Metrics #
###########


class _DummyMetric(BaseMetric):
    """Simple metric that can be flagged to simulate a skip."""

    def __init__(self, name="dummy", should_skip=False):
        self.name = name
        self.should_skip = should_skip
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5

    def measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        self.success = True

    def is_successful(self) -> bool:
        return bool(self.success)


class _DummyTaskCompletionMetric(TaskCompletionMetric):
    """Metric used to toggle the 'has_task_completion' path."""

    def __init__(self, name="tc"):
        self.name = name
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5

    def measure(self, test_case, *_args, **_kwargs):
        self.success = True

    def is_successful(self) -> bool:
        return bool(self.success)


class _SleepyMetric(BaseMetric):
    """
    Test stub that can sleep in both sync and async paths.

    Args:
        name: display name
        sleep_s: seconds to sleep (None/0 means no sleep)
        should_skip: mark as skipped instead of evaluating
        succeed: whether to set success=True after sleep, the default is False
    """

    def __init__(
        self,
        name: str = "sleepy",
        *,
        sleep_s: float | None = None,
        should_skip: bool = False,
        succeed: bool = False,
    ):
        self.name = name
        self.sleep_s = sleep_s
        self.should_skip = should_skip
        self.succeed = succeed

        # required BaseMetric fields
        self.skipped = False
        self.error = None
        self.success = False
        self.threshold = 0.5
        self.score = None
        self.reason = None

    def measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.success = bool(self.succeed)

    async def a_measure(self, test_case, *_args, **_kwargs):
        if self.should_skip:
            self.skipped = True
            return
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        self.success = bool(self.succeed)

    def is_successful(self) -> bool:
        return bool(self.success)


class _PerAttemptTimeoutMetric(BaseMetric):
    """
    A metric that intentionally exceeds the per-attempt timeout budget to trigger
    Tenacity retries. Works in both sync and async executor paths.

    Use:
      set sleep_s > per-attempt timeout
    """

    threshold = 0.0

    def __init__(self, *, sleep_s: float = 10.0):
        self.sleep_s = float(sleep_s)
        self.name = "_PerAttemptTimeoutMetric"

    # BaseMetric.measure is wrapped with run_sync_with_timeout
    def measure(self, test_case, **kwargs) -> float:
        retry = create_retry_decorator(PS.OPENAI)

        @retry
        def slow_op():
            # run_sync_with_timeout() in the retry layer enforces the per-attempt timeout
            time.sleep(self.sleep_s)
            return 1.0

        return slow_op()

    # BaseMetric.a_measure is wrapped with asyncio.wait_for
    async def a_measure(self, test_case, **kwargs) -> float:
        retry = create_retry_decorator(PS.OPENAI)

        @retry
        async def slow_op():
            # resolve_effective_attempt_timeout() will bound asyncio.wait_for(...) around this
            await asyncio.sleep(self.sleep_s)
            return 1.0

        return await slow_op()

    # required by BaseMetric
    def is_successful(self) -> bool:
        return False


#########
# Spans #
#########


class _FakeSpan:
    def __init__(self, *, input=None, output=None, metrics=None, children=None):
        self.input = input
        self.output = output
        self.expected_output = None
        self.context = None
        self.retrieval_context = None
        self.tools_called = None
        self.expected_tools = None
        self.metrics = metrics or []
        self.children = children or []
        self.status = TraceSpanStatus.SUCCESS
        self.error = None


##########
# Traces #
##########


class _FakeTrace:
    def __init__(
        self, *, input=None, output=None, metrics=None, root_span=None
    ):
        self.input = input
        self.output = output
        self.expected_output = None
        self.context = None
        self.retrieval_context = None
        self.tools_called = None
        self.expected_tools = None
        self.metrics = metrics or []
        self.root_spans = [root_span] if root_span else []
        self.status = TraceSpanStatus.SUCCESS
        self.error = None
        self.uuid = "trace-uuid"


################
# Optimization #
################


class _DummyRewriter:
    """
    Minimal object satisfying the PromptRewriterProtocol at runtime.
    Used to verify set_rewriter/get_rewriter wiring.
    """

    def rewrite(self, **kwargs):
        # Just return the original prompt unmodified
        return kwargs["old_prompt"]

    async def a_rewrite(self, **kwargs):
        return kwargs["old_prompt"]


class SuffixRewriter:
    """Rewriter that appends a suffix to the prompt text."""

    def __init__(self, suffix: str = " CHILD") -> None:
        self.suffix = suffix
        self.calls = []
        self.a_calls = []

    def rewrite(self, *, module_id, old_prompt, feedback_text, **kwargs):
        self.calls.append((module_id, old_prompt, feedback_text))
        return Prompt(
            text_template=(old_prompt.text_template or "") + self.suffix
        )

    async def a_rewrite(
        self, *, module_id, old_prompt, feedback_text, **kwargs
    ):
        self.a_calls.append((module_id, old_prompt, feedback_text))
        return self.rewrite(
            module_id=module_id,
            old_prompt=old_prompt,
            feedback_text=feedback_text,
        )


class AddBetterRewriter:
    def rewrite(
        self, *, module_id: ModuleId, old_prompt: Prompt, feedback_text: str
    ) -> Prompt:
        return Prompt(
            text_template=(
                (old_prompt.text_template or "") + " BETTER"
            ).strip(),
            messages_template=old_prompt.messages_template,
            model_settings=old_prompt.model_settings,
            output_type=old_prompt.output_type,
            output_schema=old_prompt.output_schema,
        )


class DummyRunner:
    """
    Minimal runner used to verify set_runner wiring.
    """

    def __init__(self):
        self.model_callback = None
        self.status_callback = None

    def execute(self, *, prompt, goldens):
        raise NotImplementedError

    async def a_execute(self, *, prompt, goldens):
        raise NotImplementedError


class DummyRunnerForOptimize:
    """
    Runner that simulates a completed optimization run.
    """

    def __init__(self):
        self.model_callback = None
        self.status_callback = None
        self.last_execute_args = None

    def execute(self, *, prompt, goldens):
        self.last_execute_args = (prompt, goldens)

        # Simulate an "optimized" best prompt
        best = Prompt(text_template="optimized")

        # Minimal but valid OptimizationResult-like payload
        report = {
            "optimization_id": "opt-123",
            "best_id": "best",
            "accepted_iterations": [],
            "pareto_scores": {"best": [1.0]},
            "parents": {"best": None},
            "prompt_configurations": {
                "best": {
                    "parent": None,
                    "prompts": {
                        # Arbitrary module id; just needs to be a string key
                        "module-1": {
                            "type": "TEXT",  # coerces into PromptType / Literal
                            "text_template": "optimized",
                        }
                    },
                }
            },
        }

        return best, report

    async def a_execute(self, *, prompt, goldens):
        raise AssertionError("a_execute should not be called in sync optimize")


class SyncDummyRunner:
    """
    Runner used to test _run_optimization(sync path).
    """

    def __init__(self):
        self.execute_calls = 0
        self.a_execute_calls = 0

    def execute(self, *, prompt, goldens):
        self.execute_calls += 1
        return prompt, {
            "optimization_id": "sync-id",
            "best_id": "root",
            "accepted_iterations": [],
            "pareto_scores": {"root": [1.0]},
            "parents": {"root": None},
        }

    async def a_execute(self, *, prompt, goldens):
        self.a_execute_calls += 1
        return prompt, {
            "optimization_id": "async-id",
            "best_id": "root",
            "accepted_iterations": [],
            "pareto_scores": {"root": [1.0]},
            "parents": {"root": None},
        }


class AsyncDummyRunner:
    """
    Runner used to test _run_optimization(async path).
    """

    def __init__(self):
        self.execute_calls = 0
        self.a_execute_calls = 0

    def execute(self, *, prompt, goldens):
        self.execute_calls += 1
        raise AssertionError(
            "execute() should not be called when run_async=True"
        )

    async def a_execute(self, *, prompt, goldens):
        self.a_execute_calls += 1
        return prompt, {
            "optimization_id": "opt-async",
            "best_id": "root",
            "accepted_iterations": [],
            "pareto_scores": {"root": [1.0]},
            "parents": {"root": None},
        }


class DummyProgress:
    """
    Tiny stub for rich.progress.Progress used to test _on_status.
    Records update / advance calls.
    """

    def __init__(self):
        self.records = []

    def update(self, task_id, **kwargs):
        self.records.append(("update", task_id, kwargs))

    def advance(self, task_id, amount):
        self.records.append(("advance", task_id, {"amount": amount}))


class StubScoringAdapter:
    """
    Minimal scoring adapter stub for exercising GEPARunner and other
    single-module optimization runners.

    - score_on_pareto / minibatch_score:
        returns higher scores for prompts whose text contains "CHILD"
        so that "improved" children can be accepted.
    """

    def __init__(self) -> None:
        self.pareto_calls = []
        self.a_pareto_calls = []
        self.feedback_calls = []
        self.a_feedback_calls = []
        self.score_calls = []
        self.a_score_calls = []

    def _get_prompt_text(self, prompt_configuration):
        if not getattr(prompt_configuration, "prompts", None):
            return ""
        # For GEPA/MIPRO we expect a single module id in `prompts`.
        prompt = next(iter(prompt_configuration.prompts.values()))
        return (prompt.text_template or "").strip()

    def score_on_pareto(self, prompt_configuration, d_pareto):
        self.pareto_calls.append((prompt_configuration, list(d_pareto)))
        txt = self._get_prompt_text(prompt_configuration)
        return [1.0] if "CHILD" in txt else [0.5]

    async def a_score_on_pareto(self, prompt_configuration, d_pareto):
        self.a_pareto_calls.append((prompt_configuration, list(d_pareto)))
        return self.score_on_pareto(prompt_configuration, d_pareto)

    def minibatch_feedback(self, prompt_configuration, module_id, minibatch):
        self.feedback_calls.append(
            (prompt_configuration, module_id, list(minibatch))
        )
        return "feedback"

    async def a_minibatch_feedback(
        self, prompt_configuration, module_id, minibatch
    ):
        self.a_feedback_calls.append(
            (prompt_configuration, module_id, list(minibatch))
        )
        return "feedback"

    def minibatch_score(self, prompt_configuration, minibatch):
        self.score_calls.append((prompt_configuration, list(minibatch)))
        txt = self._get_prompt_text(prompt_configuration)
        return 1.0 if "CHILD" in txt else 0.5

    async def a_minibatch_score(self, prompt_configuration, minibatch):
        self.a_score_calls.append((prompt_configuration, list(minibatch)))
        return self.minibatch_score(prompt_configuration, minibatch)


##################
# File I/O stubs #
##################


class RecordingFile(io.StringIO):
    """
    Test stub that records flush() calls and exposes a fake fileno(),
    used to verify that we call flush() and os.fsync(fd) correctly.
    """

    def __init__(self):
        super().__init__()
        self.flushed = False
        self.closed_flag = False
        # Arbitrary fake file descriptor; tests only check identity equality
        self._fd = 42

    def flush(self):
        self.flushed = True
        return super().flush()

    def fileno(self):
        return self._fd

    def close(self):
        self.closed_flag = True
        return super().close()


class RecordingPortalockerLock:
    """
    Minimal drop-in for portalocker.Lock used in tests.

    It always returns a new RecordingFile and exposes the most recently
    created one via the class attribute `last_file` so tests can assert on it.
    """

    last_file = None

    def __init__(self, *args, **kwargs):
        self.file = RecordingFile()
        RecordingPortalockerLock.last_file = self.file

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc, tb):
        self.file.close()
