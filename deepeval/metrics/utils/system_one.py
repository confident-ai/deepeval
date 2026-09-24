"""The whole-chain System One path for built-in metrics (`system_one` eval
mode).

A built-in LLM-as-a-judge metric normally runs a chain: the LLM extracts
items, a judge decides about each, code turns the decisions into a score and
the LLM writes a reason. Under `system_one` eval mode the metric instead
describes itself as a `SystemOneEvalSpec`: which test case fields to send as
state and which bounded questions (`Noul` / `Score` / `Choice`, the same
primitives `JevEval` uses) to ask about them. One `decide()` answers them all,
the score is the weighted mean of the answers mapped onto `[0, 1]`, and the
reason is deterministic text built from the answers. No LLM call is made on
this path.

`run_system_one_eval` is the bridge each metric's `measure` / `a_measure`
calls first. It returns `True` when Jev decided the measure (the metric is
fully populated and can return), and `False` when the mode is not
`system_one` or the metric has no whole-chain form, so the metric runs its
usual chain. Under `system_one` nothing falls back: the user asked for Jev,
so a context overflow raises telling them to switch back to `llm`, any other
Jev error surfaces unchanged, and a low-confidence result is kept with
`metric.confidence` reporting it.

A metric opts in by overriding `_system_one_eval_spec(test_case)` on the base
class; metrics that never do (DAG, a user's own subclass) run as `hybrid`
under `system_one` mode (see `effective_eval_mode`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import Annotated

from deepeval.errors import DeepEvalError
from deepeval.models.system_one.limits import (
    SystemOneContextLimitError,
    check_context_budget,
)
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MultiTurnParams,
    SingleTurnParams,
)

from .decision import (
    EVAL_MODE_ENV_VAR,
    EvalMode,
    _accrue,
    _jsonable,
    _record_confidence,
    _unwrap,
    context_limit_error,
    effective_eval_mode,
)
from .verbose import construct_verbose_logs

if TYPE_CHECKING:
    from deepeval.metrics.jev_eval.questions import (
        JevQuestion,
        QuestionOutcome,
    )
    from deepeval.models.system_one.schema import ChoiceAnswer


###############################################
# Spec
###############################################


@dataclass
class SystemOneEvalSpec:
    """What a metric sends to Jev under `system_one` eval mode.

    `evaluation_params` are the test case fields that become the state (for a
    `ConversationalTestCase` the turns are always included). `extra_state`
    adds what is not a test case field (a trace, the available tools, a
    metric argument), each under its own top-level key the questions can
    name. `questions` are the decision points; each maps onto `[0, 1]` by the
    rules in `deepeval.metrics.jev_eval.utils`, and the score is their
    weighted mean. There is no reason prompt: the reason is built from the
    answers.
    """

    evaluation_params: Sequence[Union[SingleTurnParams, MultiTurnParams]]
    questions: Sequence["JevQuestion"]
    extra_state: Dict[str, Any] = field(default_factory=dict)


_TRACE_SPAN_KEYS = (
    "name",
    "type",
    "description",
    "input",
    "output",
    "error",
    "model",
    "retrieval_context",
    "context",
    "expected_output",
    "tools_called",
    "expected_tools",
    "available_tools",
    "agent_handoffs",
)


def compact_trace(trace: Any) -> Any:
    """A trace as Jev reads it: each span's name, type, inputs, outputs and
    tool calls, nested under `children`. Token counts, costs, prompts and
    integration details are dropped; they say nothing about whether the
    agent did the job and only eat into Jev's context."""
    if not isinstance(trace, dict):
        return trace
    span: Dict[str, Any] = {}
    for key in _TRACE_SPAN_KEYS:
        value = trace.get(key)
        if value is not None and value != [] and value != "":
            span[key] = value
    children = [compact_trace(c) for c in trace.get("children") or []]
    if children:
        span["children"] = children
    return _jsonable(span)


_QUESTIONS_ADAPTER: Optional[TypeAdapter] = None


def parse_questions(rendered: str) -> List["JevQuestion"]:
    """Turn a rendered `_experimental_system_one_questions` template (a JSON
    array of `Noul` / `Score` / `Choice` objects discriminated on `type`)
    into question objects. Keeping the questions in the template bundle means
    they can be found, reviewed and overridden like any other prompt."""
    global _QUESTIONS_ADAPTER
    from deepeval.metrics.jev_eval.questions import Choice, Noul, Score

    if _QUESTIONS_ADAPTER is None:
        _QUESTIONS_ADAPTER = TypeAdapter(
            List[
                Annotated[
                    Union[Noul, Score, Choice], Field(discriminator="type")
                ]
            ]
        )
    try:
        return _QUESTIONS_ADAPTER.validate_python(json.loads(rendered))
    except (json.JSONDecodeError, ValidationError) as e:
        raise DeepEvalError(
            f"_experimental_system_one_questions must render to a JSON array "
            f"of Noul / Score / Choice questions: {e}"
        ) from e


###############################################
# Runner
###############################################


def _prepare(
    metric: Any, test_case: Any
) -> Optional[Tuple[SystemOneEvalSpec, Dict[str, Any], Dict[str, Any]]]:
    """Everything needed for the one `decide()` call, or `None` when this
    measure should not go through Jev at all."""
    mode = effective_eval_mode(metric)
    if mode is not EvalMode.SYSTEM_ONE:
        return None
    spec = metric._system_one_eval_spec(test_case)
    if spec is None:
        if getattr(metric, "model", None) is None:
            # A wired metric under `system_one` builds no LLM, so there is no
            # other chain to run.
            raise DeepEvalError(
                f"{EVAL_MODE_ENV_VAR}={mode} runs "
                f"{getattr(metric, '__name__', type(metric).__name__)} on "
                f"System One, which cannot judge this test case. Switch it "
                f'back to the LLM with `eval_mode="{EvalMode.LLM}"` or '
                f"`deepeval set-eval-mode {EvalMode.LLM}`."
            )
        return None
    if getattr(metric, "system_one_model", None) is None:
        raise DeepEvalError(
            f"{EVAL_MODE_ENV_VAR}={mode} runs {type(metric).__name__} on a "
            f"System One model, but none is configured. Set TYPESAFE_API_KEY "
            f"or switch back with `deepeval set-eval-mode {EvalMode.LLM}`."
        )

    from deepeval.metrics.jev_eval.utils import (
        build_questions,
        construct_multi_turn_state,
        construct_single_turn_state,
    )

    if isinstance(test_case, ConversationalTestCase):
        params = [
            p for p in spec.evaluation_params if isinstance(p, MultiTurnParams)
        ]
        state = construct_multi_turn_state(params, test_case)
    elif isinstance(test_case, LLMTestCase):
        params = [
            p for p in spec.evaluation_params if isinstance(p, SingleTurnParams)
        ]
        state = construct_single_turn_state(params, test_case)
    else:
        return None
    if not state.get("test_case", True):
        state.pop("test_case")
    for key, value in spec.extra_state.items():
        if value is not None:
            state[key] = _jsonable(value)
    return spec, state, build_questions(spec.questions)


def _decide(metric: Any, state: Any, questions: Any):
    # The runner sends the whole test case, the largest state any metric
    # builds, so it checks the budget itself rather than relying on the
    # model to (a custom `DeepEvalBaseSystemOneModel` may not).
    try:
        check_context_budget(state, questions)
        return metric.system_one_model.decide(state, questions)
    except Exception as e:  # noqa: BLE001 - only the overflow is rewrapped
        inner = _unwrap(e)
        if isinstance(inner, SystemOneContextLimitError):
            raise context_limit_error(metric, inner) from inner
        raise


async def _a_decide(metric: Any, state: Any, questions: Any):
    try:
        check_context_budget(state, questions)
        return await metric.system_one_model.a_decide(state, questions)
    except Exception as e:  # noqa: BLE001 - only the overflow is rewrapped
        inner = _unwrap(e)
        if isinstance(inner, SystemOneContextLimitError):
            raise context_limit_error(metric, inner) from inner
        raise


def _populate(
    metric: Any, spec: SystemOneEvalSpec, answers: Any, cost: Any
) -> None:
    """Book the answers and fill in every field a measure reports."""
    from deepeval.metrics.jev_eval.utils import (
        aggregate,
        aggregate_strict,
        mark_strict,
        min_confidence,
        outcomes_from_answers,
    )

    _accrue(metric, cost)
    _record_confidence(metric, answers)
    outcomes = outcomes_from_answers(spec.questions, answers)
    confidence = min_confidence(outcomes)

    if metric.strict_mode:
        outcomes = mark_strict(spec.questions, outcomes)
        metric.score = aggregate_strict(outcomes)
    else:
        metric.score = aggregate(outcomes)
    metric._system_one_outcomes = outcomes
    metric.score_breakdown = [o.model_dump() for o in outcomes]
    metric.confidence = confidence
    metric.reason = (
        format_system_one_reason(metric, outcomes)
        if metric.include_reason
        else None
    )
    metric.success = metric.is_successful()
    metric.evaluation_model = metric.system_one_model.get_model_name()
    metric.verbose_logs = construct_verbose_logs(
        metric,
        steps=[
            f"Decided by System One ({metric.evaluation_model}); no LLM "
            f"call was made.",
            f"Questions:\n{format_outcomes_for_logs(outcomes)}",
            f"Score: {metric.score}\nConfidence: {confidence}\n"
            f"Reason: {metric.reason}",
        ],
    )


def run_system_one_eval(metric: Any, test_case: Any) -> bool:
    """Decide the whole measure with Jev. See the module docstring."""
    prepared = _prepare(metric, test_case)
    if prepared is None:
        return False
    spec, state, questions = prepared
    _populate(metric, spec, *_decide(metric, state, questions))
    return True


async def a_run_system_one_eval(metric: Any, test_case: Any) -> bool:
    """Async counterpart of `run_system_one_eval`."""
    prepared = _prepare(metric, test_case)
    if prepared is None:
        return False
    spec, state, questions = prepared
    _populate(metric, spec, *(await _a_decide(metric, state, questions)))
    return True


###############################################
# Deterministic reasons
###############################################
#
# Nothing here calls a model. The reason states who decided, how confident
# the least decisive answer was, and what each answer was in words and
# numbers, so a reader can see exactly where the score came from.


def _judge_name(metric: Any) -> str:
    model = getattr(metric, "system_one_model", None)
    return model.get_model_name() if model is not None else "System One"


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _outcome_line(index: int, outcome: "QuestionOutcome") -> str:
    from deepeval.metrics.jev_eval.utils import verbalise_outcome

    words = verbalise_outcome(outcome)
    if outcome.type == "noul":
        numbers = f"P(yes)={_fmt(outcome.probabilities.get('true'))}"
    elif outcome.type == "score":
        numbers = f"expected level={_fmt(outcome.value)} of 1.00"
    elif not outcome.applicable:
        numbers = "not applicable"
    else:
        top = max(
            outcome.probabilities.items(), key=lambda kv: kv[1], default=None
        )
        numbers = f"P={_fmt(top[1])}" if top else "P=n/a"
    line = f"{index}. {outcome.question} -> {words} ({numbers}"
    if outcome.confidence is not None:
        line += f", confidence={_fmt(outcome.confidence)}"
    if outcome.weight != 1.0:
        line += f", weight={outcome.weight:g}"
    line += ")"
    if outcome.passed is not None:
        line += f" strict={'pass' if outcome.passed else 'fail'}"
    return line


def format_system_one_reason(
    metric: Any, outcomes: Sequence["QuestionOutcome"]
) -> str:
    """The reason for a measure Jev decided: judge, confidence, one line per
    question, and how the score follows from them."""
    from deepeval.metrics.jev_eval.utils import min_confidence

    confidence = min_confidence(outcomes)
    header = f"Decided by {_judge_name(metric)}"
    if confidence is not None:
        header += f", minimum confidence {confidence:.2f}"
    header += "."
    lines = [header]
    lines += [_outcome_line(i, o) for i, o in enumerate(outcomes, start=1)]

    applicable = [o for o in outcomes if o.applicable]
    score = getattr(metric, "score", None)
    if getattr(metric, "strict_mode", False):
        passed = all(o.passed for o in applicable) if applicable else True
        lines.append(
            f"Score: {_fmt(score)} (strict mode: "
            f"{'every applicable question passed' if passed else 'at least one applicable question failed'})."
        )
    elif not applicable:
        lines.append(
            f"Score: {_fmt(score)} (no question applied to this test case)."
        )
    else:
        lines.append(
            f"Score: {_fmt(score)} (weighted mean of {len(applicable)} "
            f"applicable question{'s' if len(applicable) != 1 else ''})."
        )
    return "\n".join(lines)


def format_classification_reason(
    classifier: Any, answer: "ChoiceAnswer"
) -> str:
    """The reason for a label Jev chose: the label, its probability and
    confidence, then every alternative in descending order."""
    chosen_p = answer.probabilities.get(answer.choice, 0.0)
    text = (
        f'Decided by {_judge_name(classifier)}. Label "{answer.choice}" '
        f"(P={chosen_p:.2f}, confidence={answer.confidence:.2f})."
    )
    alternatives = sorted(
        (
            (name, p)
            for name, p in answer.probabilities.items()
            if name != answer.choice
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if alternatives:
        text += " Alternatives: " + ", ".join(
            f"{name} {p:.2f}" for name, p in alternatives
        )
        text += "."
    return text


def format_outcomes_for_logs(outcomes: Sequence["QuestionOutcome"]) -> str:
    from deepeval.metrics.jev_eval.utils import (
        format_outcomes_for_logs as _format,
    )

    return _format(outcomes)
