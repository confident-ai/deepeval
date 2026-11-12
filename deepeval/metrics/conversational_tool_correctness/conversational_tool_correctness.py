import re
import unicodedata
from typing import Any, Dict, List, Optional

from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase, ToolCall


def _normalized(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = re.sub(r"\s+", "", value).casefold()
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return sorted((_normalized(item) for item in value), key=repr)
    return value


def _is_free_text(value: Any) -> bool:
    """A description in words, such as a reason, a note or an address, as
    opposed to a single value such as a code, a number or a date."""
    return isinstance(value, str) and len(value.split()) >= 3


def _words(value: str) -> set:
    plain = unicodedata.normalize("NFKD", value.casefold())
    return set(re.findall(r"[a-z]{3,}", plain))


def _close_text(actual: Any, expected: str) -> bool:
    """Free text is said in the agent's own words: it matches when every
    number in it is the same and most of the expected words are there."""
    if not isinstance(actual, str):
        return False
    if re.findall(r"\d+", actual) != re.findall(r"\d+", expected):
        return False
    wanted = _words(expected)
    return not wanted or len(wanted & _words(actual)) / len(wanted) >= 0.5


def _given(value: Any) -> bool:
    return value not in (None, "", [], {})


def _contains(actual: Any, expected: Any, partial_lists: bool) -> bool:
    """True when `actual` carries everything `expected` asks for.

    Dictionaries match as subsets, so arguments the expected call leaves out
    are free to take any value. With `partial_lists`, a list matches when it
    holds every expected item, whatever else it holds.
    """
    if isinstance(expected, dict) and set(expected) == {"any_of"}:
        # Any one of several accepted values, such as the same contents said
        # in the caller's language or in English.
        return any(
            _contains(actual, option, partial_lists)
            for option in expected["any_of"]
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual
            and _given(actual[key])
            and _contains(actual[key], item, partial_lists)
            for key, item in expected.items()
        )
    if _is_free_text(expected):
        return _close_text(actual, expected)
    if partial_lists and isinstance(expected, (list, tuple, set)):
        if not isinstance(actual, (list, tuple, set)):
            return False
        held = [_normalized(item) for item in actual]
        return all(_normalized(item) in held for item in expected)
    return _normalized(actual) == _normalized(expected)


def _matches(
    actual: ToolCall,
    expected: ToolCall,
    partial_lists: bool = False,
    lookup: bool = False,
) -> bool:
    if actual.name != expected.name:
        return False
    given = actual.input_parameters or {}
    wanted = expected.input_parameters or {}
    if lookup:
        # A lookup finds the record by any of its keys, so it only has to
        # get right the ones it was given, and be given at least one.
        keys = [key for key in wanted if _given(given.get(key))]
        return bool(keys) and all(
            _contains(given[key], wanted[key], partial_lists) for key in keys
        )
    return _contains(given, wanted, partial_lists)


def _describe(tool: ToolCall) -> str:
    arguments = ", ".join(
        f"{key}={value!r}"
        for key, value in (tool.input_parameters or {}).items()
    )
    return f"{tool.name}({arguments})"


class ConversationalToolCorrectnessMetric(BaseVoiceMetric):
    """Checks the tool calls made across a whole conversation.

    Every expected call must have been made with the expected arguments, and
    no forbidden call may have been made. An expected or forbidden call only
    constrains the arguments it lists.

    Free text, such as a reason, matches when most of its words are there.
    A call named in `lookup_tools` only reads a record, so it matches when
    the keys it was given are right and at least one was given.
    """

    def __init__(
        self,
        expected_tools: List[ToolCall],
        forbidden_tools: Optional[List[ToolCall]] = None,
        should_consider_ordering: bool = False,
        lookup_tools: Optional[List[str]] = None,
        threshold: Optional[float] = 1.0,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.expected_tools = expected_tools
        self.forbidden_tools = forbidden_tools or []
        self.should_consider_ordering = should_consider_ordering
        self.lookup_tools = set(lookup_tools or [])

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        called = [
            tool
            for turn in test_case.turns
            if turn.role == "assistant"
            for tool in turn.tools_called or []
        ]
        if not self.expected_tools and not self.forbidden_tools:
            return None, "No expected or forbidden tool calls were given.", {}

        used: List[int] = []
        missing: List[str] = []
        out_of_order: List[str] = []
        for expected in self.expected_tools:
            start = (
                (used[-1] + 1 if used else 0)
                if self.should_consider_ordering
                else 0
            )
            index = next(
                (
                    i
                    for i in range(start, len(called))
                    if i not in used
                    and _matches(
                        called[i],
                        expected,
                        lookup=expected.name in self.lookup_tools,
                    )
                ),
                None,
            )
            if index is not None:
                used.append(index)
            elif self.should_consider_ordering and any(
                _matches(
                    tool, expected, lookup=expected.name in self.lookup_tools
                )
                for tool in called
            ):
                out_of_order.append(_describe(expected))
            else:
                missing.append(_describe(expected))

        forbidden_called = [
            _describe(tool)
            for tool in called
            if any(
                _matches(tool, forbidden, partial_lists=True)
                for forbidden in self.forbidden_tools
            )
        ]
        matched = len(used)
        total = len(self.expected_tools)
        score = matched / total if total else 1.0
        if forbidden_called:
            score = 0.0

        breakdown: Dict = {
            "expected": total,
            "matched": matched,
            "missing": missing,
            "out_of_order": out_of_order,
            "forbidden_called": forbidden_called,
            "called": [_describe(tool) for tool in called],
        }
        if forbidden_called:
            reason = (
                "The agent made a call it must not make: "
                + "; ".join(forbidden_called)
                + "."
            )
        elif missing or out_of_order:
            problems = [f"missing {call}" for call in missing] + [
                f"out of order {call}" for call in out_of_order
            ]
            reason = (
                f"{matched} of {total} expected tool calls were made correctly: "
                + "; ".join(problems)
                + "."
            )
        else:
            reason = (
                f"All {total} expected tool calls were made with the expected "
                "arguments, and no forbidden call was made."
            )
        return score, reason, breakdown

    @property
    def __name__(self):
        return "Conversational Tool Correctness"
