"""Batched System One for `evaluate()`: every metric Jev judges on a test
case, asked in as few requests as the token budget allows.

Under `system_one` a metric's own `measure` sends one request per metric
(`run_system_one_eval`), re-sending the same test case each time. Jev
evaluates every question in a request independently against one shared
state, so the executors instead collect a test case's `system_one` metrics,
merge their states and questions into one request, and hand each metric
back its own answers and share of the cost. Scores, reasons and confidence
are filled in by the same `_populate` a lone measure uses.

A metric is batched when its effective eval mode is `system_one` (so an
`eval_mode=` override on the metric is honoured), it has a whole-metric
form for the test case, and it has no cached result. With fewer than two
such metrics there is nothing to share, and every metric runs its own
`measure`.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from deepeval.config.eval_mode import EvalMode
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.base_metric import BaseConversationalMetric, BaseMetric
from deepeval.metrics.indicator import start_measure
from deepeval.models.system_one.limits import (
    SystemOneContextLimitError,
    check_context_budget,
)
from deepeval.models.system_one.schema import (
    SystemOneAnswers,
    SystemOneQuestion,
)
from deepeval.models.utils import EvaluationCost
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.test_run.cache import Cache, CachedTestCase
from deepeval.utils import add_pbar, format_error_text, update_pbar

from .decision import Metric, _unwrap, context_limit_error, effective_eval_mode
from .system_one import SystemOneEvalSpec, _populate, _request_for
from .test_case import prepare_measure

###############################################
# Requests
###############################################


@dataclass
class _Entry:
    metric: Metric
    spec: SystemOneEvalSpec
    state: Dict[str, Any]
    questions: Dict[str, SystemOneQuestion]


@dataclass
class _Request:
    """One Jev call. Questions are keyed `m{j}.{key}` so each entry's answers
    can be picked back out; keys are never sent to the model."""

    entries: List[_Entry]
    state: Dict[str, Any]

    @property
    def model(self) -> Any:
        return self.entries[0].metric.system_one_model

    @property
    def questions(self) -> Dict[str, SystemOneQuestion]:
        return {
            f"m{j}.{key}": question
            for j, entry in enumerate(self.entries)
            for key, question in entry.questions.items()
        }

    def admit(self, entry: _Entry) -> bool:
        """Add `entry` when it asks the same Jev model, its state merges
        without conflict and the merged request still fits Jev's budget."""
        model_name = entry.metric.system_one_model.get_model_name()
        if model_name != self.model.get_model_name():
            return False
        state = _merge_states(self.state, entry.state)
        if state is None:
            return False
        merged = _Request(self.entries + [entry], state)
        try:
            check_context_budget(merged.state, merged.questions)
        except SystemOneContextLimitError:
            return False
        self.entries, self.state = merged.entries, merged.state
        return True

    def answers_for(
        self, answers: SystemOneAnswers, j: int
    ) -> SystemOneAnswers:
        prefix = f"m{j}."

        def pick(by_key: Dict[str, Any]) -> Dict[str, Any]:
            return {
                key[len(prefix) :]: answer
                for key, answer in by_key.items()
                if key.startswith(prefix)
            }

        return SystemOneAnswers(
            nouls=pick(answers.nouls),
            choices=pick(answers.choices),
            scores=pick(answers.scores),
        )


def _merge_states(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """`a` and `b` as one state, or `None` when they give the same top-level
    key different values. `test_case` fields and each turn's fields always
    merge: both states were built from the same test case."""
    merged = dict(a)
    for key, value in b.items():
        if key == "test_case":
            merged[key] = {**a.get(key, {}), **value}
        elif key == "turns" and key in merged:
            merged[key] = [
                {**mine, **theirs} for mine, theirs in zip(merged[key], value)
            ]
        elif key in merged and merged[key] != value:
            return None
        else:
            merged[key] = value
    return merged


def _pack(entries: Sequence[_Entry]) -> List[_Request]:
    requests: List[_Request] = []
    for entry in entries:
        if not any(request.admit(entry) for request in requests):
            requests.append(_Request([entry], entry.state))
    return requests


###############################################
# Selection and setup
###############################################


def _candidates(
    metrics: Sequence[Any],
    test_case: Any,
    cached_test_case: Optional[CachedTestCase],
) -> List[Metric]:
    if isinstance(test_case, LLMTestCase):
        metric_type = BaseMetric
    elif isinstance(test_case, ConversationalTestCase):
        metric_type = BaseConversationalMetric
    else:
        return []
    # Every whole-metric spec returns `None` for a multimodal test case, so
    # none of its metrics could be batched.
    if test_case.multimodal:
        return []
    return [
        metric
        for metric in metrics
        if isinstance(metric, metric_type)
        and effective_eval_mode(metric) is EvalMode.SYSTEM_ONE
        and Cache.get_metric_data(metric, cached_test_case) is None
    ]


def _fail(
    metric: Metric,
    exc: BaseException,
    ignore_errors: bool,
    skip_on_missing_params: bool,
) -> None:
    """Record a failed measure the way the executors do for a metric's own
    `measure`."""
    if isinstance(exc, MissingTestCaseParamsError) and skip_on_missing_params:
        metric.skipped = True
        metric.error = None
        metric.success = None
    elif ignore_errors:
        metric.error = format_error_text(exc)
        metric.success = False
    else:
        raise exc


def _setup(
    candidates: Sequence[Metric],
    test_case: Any,
    *,
    async_mode: bool,
    ignore_errors: bool,
    skip_on_missing_params: bool,
) -> Tuple[List[Metric], List[_Entry]]:
    """Run each candidate's pre-measure setup and build its Jev request.
    Returns the metrics this batch now owns (including ones whose setup
    failed) and the entries to send. A metric whose spec declines the test
    case is left for its own `measure`."""
    handled: List[Metric] = []
    entries: List[_Entry] = []
    for metric in candidates:
        metric.skipped = False
        try:
            prepare_measure(metric, test_case)
            spec = metric._system_one_eval_spec(test_case)
            prepared = spec and _request_for(metric, test_case, spec)
        except Exception as e:  # noqa: BLE001 - recorded per error config
            _fail(metric, e, ignore_errors, skip_on_missing_params)
            handled.append(metric)
            continue
        if not prepared:
            continue
        start_measure(metric, async_mode=async_mode)
        entries.append(_Entry(metric, *prepared))
        handled.append(metric)
    return handled, entries


###############################################
# Answers and cost
###############################################


def _apportion(total: Optional[int], weights: Sequence[int]) -> List[Any]:
    """Split an integer `total` by `weights` so the parts sum back to it."""
    if total is None:
        return [None] * len(weights)
    whole = sum(weights)
    bounds = [0]
    running = 0
    for weight in weights:
        running += weight
        bounds.append(round(total * running / whole))
    return [hi - lo for lo, hi in zip(bounds, bounds[1:])]


def _split_cost(cost: Any, weights: Sequence[int]) -> List[Any]:
    """Each entry's share of a request's cost, by its share of the
    questions. Token counts ride along so `accrue_token_usage` stays exact."""
    if cost is None:
        return [None] * len(weights)
    whole = sum(weights)
    fractions = [weight / whole for weight in weights]
    if not isinstance(cost, EvaluationCost):
        return [cost * fraction for fraction in fractions]
    return [
        EvaluationCost(cost.value * fraction, input_tokens, output_tokens)
        for fraction, input_tokens, output_tokens in zip(
            fractions,
            _apportion(cost.input_tokens, weights),
            _apportion(cost.output_tokens, weights),
        )
    ]


def _request_error(request: _Request, exc: BaseException) -> BaseException:
    inner = _unwrap(exc)
    if (
        isinstance(inner, SystemOneContextLimitError)
        and len(request.entries) == 1
    ):
        return context_limit_error(request.entries[0].metric, inner)
    return exc


def _settle(request: _Request, result: Any, ignore_errors: bool) -> None:
    """Hand each entry its answers and cost share, or the request's error.
    Nothing falls back to the LLM, as under `system_one` generally."""
    if isinstance(result, BaseException):
        if not isinstance(result, Exception):
            raise result
        error = _request_error(request, result)
        for entry in request.entries:
            _fail(entry.metric, error, ignore_errors, False)
        return
    answers, cost = result
    shares = _split_cost(cost, [len(e.questions) for e in request.entries])
    for j, (entry, share) in enumerate(zip(request.entries, shares)):
        _populate(
            entry.metric, entry.spec, request.answers_for(answers, j), share
        )


# The runner sends the whole test case, the largest state any metric
# builds, so it checks the budget itself rather than relying on the model to
# (a custom `DeepEvalBaseSystemOneModel` may not).
def _call(request: _Request) -> Any:
    check_context_budget(request.state, request.questions)
    return request.model.decide(request.state, request.questions)


async def _a_call(request: _Request) -> Any:
    check_context_budget(request.state, request.questions)
    return await request.model.a_decide(request.state, request.questions)


###############################################
# Progress
###############################################


def _describe(requests: Sequence[_Request]) -> str:
    metrics = [
        entry.metric for request in requests for entry in request.entries
    ]
    models = sorted({request.model.get_model_name() for request in requests})
    count = len(requests)
    return (
        f"Judging {', '.join(m.__name__ for m in metrics)} with "
        f"{', '.join(models)} ({count} request{'s' if count != 1 else ''})"
    )


@contextmanager
def _indicator(
    requests: Sequence[_Request],
    show_indicator: bool,
    progress: Optional[Progress],
) -> Iterator[None]:
    """One line for the whole batch: a child task under the test case's bar,
    or a single spinner in place of one per metric."""
    if not requests:
        yield
        return
    if progress is not None:
        metrics = sum(len(request.entries) for request in requests)
        count = len(requests)
        task = add_pbar(
            progress,
            f"    ⚡ Jev: judging {metrics} metrics in {count} "
            f"request{'s' if count != 1 else ''}",
        )
        yield
        update_pbar(progress, task)
        return
    if not show_indicator:
        yield
        return
    with Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        BarColumn(bar_width=60),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as spinner:
        description = f"✨ {_describe(requests)}"
        task = spinner.add_task(description=description, total=100)
        start = time.perf_counter()
        yield
        spinner.update(
            task,
            advance=100,
            description=f"{description} [rgb(25,227,160)]Done! "
            f"({time.perf_counter() - start:.2f}s)",
        )


###############################################
# Entry points
###############################################


def measure_system_one_batch(
    metrics: Sequence[Any],
    test_case: Any,
    *,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    cached_test_case: Optional[CachedTestCase] = None,
    show_indicator: bool = False,
    progress: Optional[Progress] = None,
) -> List[Metric]:
    """Judge every batchable metric on `test_case` with Jev and return the
    metrics it handled (judged, errored or skipped); the caller runs the
    rest as usual. Returns `[]` when fewer than two metrics qualify."""
    candidates = _candidates(metrics, test_case, cached_test_case)
    if len(candidates) < 2:
        return []
    handled, entries = _setup(
        candidates,
        test_case,
        async_mode=False,
        ignore_errors=ignore_errors,
        skip_on_missing_params=skip_on_missing_params,
    )
    requests = _pack(entries)
    with _indicator(requests, show_indicator, progress):
        for request in requests:
            try:
                result = _call(request)
            except Exception as e:  # noqa: BLE001 - settled per error config
                result = e
            _settle(request, result, ignore_errors)
    return handled


async def a_measure_system_one_batch(
    metrics: Sequence[Any],
    test_case: Any,
    *,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    cached_test_case: Optional[CachedTestCase] = None,
    show_indicator: bool = False,
    progress: Optional[Progress] = None,
) -> List[Metric]:
    """Async counterpart of `measure_system_one_batch`; requests run
    concurrently."""
    candidates = _candidates(metrics, test_case, cached_test_case)
    if len(candidates) < 2:
        return []
    handled, entries = _setup(
        candidates,
        test_case,
        async_mode=True,
        ignore_errors=ignore_errors,
        skip_on_missing_params=skip_on_missing_params,
    )
    requests = _pack(entries)
    with _indicator(requests, show_indicator, progress):
        results = await asyncio.gather(
            *(_a_call(request) for request in requests),
            return_exceptions=True,
        )
        for request, result in zip(requests, results):
            _settle(request, result, ignore_errors)
    return handled
