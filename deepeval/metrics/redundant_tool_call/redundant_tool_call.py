import json
from typing import Any, Dict, List, Optional, Set, Tuple

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.metrics.indicator import metric_progress_indicator

# A tool output shorter than this (after normalization) is not searched for in
# LLM inputs -- a short string like "OK" or "42" would match incidentally and
# report a wasteful call as consumed.
_MIN_CONSUMPTION_MATCH_LENGTH = 12


class RedundantToolCallMetric(BaseMetric):
    """Detects *wasted* tool work in an agent trace.

    Two independent, deterministic sub-signals are combined with ``min()``
    so the metric behaves as a gate -- either form of waste fails it:

    1. **Duplicate calls** -- the same ``(name, arguments)`` read-only tool
       call made more than once. The second identical call to a cacheable
       tool cannot return new information, so the work is wasted.

    2. **Unconsumed output** -- a tool span whose output text never appears
       in the input of any LLM span that is not one of its own ancestors.
       The agent fetched data and then never fed it to a model.

    Relationship to ``AgentLoopDetectionMetric``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The two metrics measure different failures and are complementary:

    * ``AgentLoopDetectionMetric`` measures **volume** -- it asks "is this
      agent stuck?" and only alarms once a call repeats
      ``repetition_threshold`` times (3 by default). It is a loop alarm.
    * This metric measures **waste** -- it alarms on the *first* redundant
      pair (threshold 1), only for tools the caller declares read-only, and
      additionally catches fetched-but-unused output. An agent that calls
      ``search_kb`` twice and finishes is not looping (ALD scores it 1.0)
      but it did burn a redundant call, which this metric reports.

    Design decisions
    ~~~~~~~~~~~~~~~~
    * **Fully deterministic** -- no LLM and no API key, so it is free to run
      as a CI gate.
    * **Read-only scoping is opt-in per tool.** Re-calling a *write* tool is
      usually intentional (sending two emails is not redundancy), so
      duplicate detection is only meaningful for tools the caller marks as
      cacheable via ``read_only_tools``. When ``read_only_tools`` is left as
      ``None`` every tool is treated as cacheable; pass an explicit list for
      any agent that performs writes.
    * **Consumption is existence-based, not time-ordered.** The trace dict
      produced by ``create_nested_spans_dict`` strips ``start_time`` and
      ``end_time``, so spans cannot be ordered in time. Rather than invent an
      ordering from traversal position (which is not chronological -- a tool
      nested under an LLM span is visited *after* it), a tool output counts as
      consumed if it appears in *any* non-ancestor LLM span's input. Ancestors
      are excluded because an invoking span's input was already fixed before
      its child tool ran, so it cannot have consumed the result.

    Limitations
    ~~~~~~~~~~~
    * **Read-only classification is user-supplied.** Side effects cannot be
      inferred from a trace, so a tool that is misdeclared as read-only will
      have its intentional repeat calls reported as redundant.
    * **Consumption uses normalized substring matching.** A tool output that
      the agent summarized, reformatted, or quoted only in part before
      passing it to the next LLM call will be reported as unconsumed even
      though it was used. An LLM-as-judge check would handle paraphrase but
      would sacrifice the deterministic, zero-cost properties.
    * **No temporal ordering.** Because timing is stripped from the trace
      dict, this metric cannot distinguish "consumed by a later call" from
      "consumed by an unrelated branch of the trace".
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        read_only_tools: Optional[List[str]] = None,
        check_duplicate_calls: bool = True,
        check_unconsumed_output: bool = True,
        threshold: float = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        if not check_duplicate_calls and not check_unconsumed_output:
            raise ValueError(
                "RedundantToolCallMetric requires at least one of "
                "`check_duplicate_calls` or `check_unconsumed_output` "
                "to be enabled."
            )
        self.read_only_tools = (
            set(read_only_tools) if read_only_tools is not None else None
        )
        self.check_duplicate_calls = check_duplicate_calls
        self.check_unconsumed_output = check_unconsumed_output
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.evaluation_model = None
        self.async_mode = False
        self.requires_trace = True

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0

        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self._calculate_metric(test_case)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric -- no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def _calculate_metric(self, test_case: LLMTestCase):
        if test_case._trace_dict is None:
            self.score = 0.0
            self.success = False
            self.reason = (
                "No trace data found. This metric requires trace "
                "data from @observe."
            )
            self.verbose_logs = ""
            return

        all_spans = self._extract_all_spans(test_case._trace_dict)
        tool_spans = [s for s in all_spans if s.get("type") == "tool"]

        dup_score, dup_reason = 1.0, "Duplicate call check skipped."
        duplicates: List[Tuple[str, int]] = []
        if self.check_duplicate_calls:
            dup_score, dup_reason, duplicates = self._score_duplicate_calls(
                tool_spans
            )

        unc_score, unc_reason = 1.0, "Unconsumed output check skipped."
        unconsumed: List[str] = []
        if self.check_unconsumed_output:
            unc_score, unc_reason, unconsumed = self._score_unconsumed_output(
                test_case._trace_dict
            )

        self.score_breakdown = {
            "duplicate_calls": dup_score,
            "unconsumed_outputs": unc_score,
            "redundant_calls": duplicates,
            "unconsumed_tools": unconsumed,
        }

        enabled_scores = []
        if self.check_duplicate_calls:
            enabled_scores.append(dup_score)
        if self.check_unconsumed_output:
            enabled_scores.append(unc_score)
        score = min(enabled_scores) if enabled_scores else 1.0

        self.score = (
            0.0 if self.strict_mode and score < self.threshold else score
        )
        self.success = self.score >= self.threshold

        reasons = []
        if self.check_duplicate_calls and dup_score < 1.0:
            reasons.append(dup_reason)
        if self.check_unconsumed_output and unc_score < 1.0:
            reasons.append(unc_reason)

        if not reasons:
            self.reason = "No redundant tool usage detected."
        else:
            self.reason = " ".join(reasons)

        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Read-only tools: "
                f"{sorted(self.read_only_tools) if self.read_only_tools is not None else 'ALL (default)'}",
                f"Tool spans found: {[s.get('name') for s in tool_spans]}",
                f"Duplicate Calls Score: {dup_score} ({dup_reason})",
                f"Unconsumed Output Score: {unc_score} ({unc_reason})",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _extract_all_spans(self, trace_dict: Optional[Dict]) -> List[Dict]:
        if not trace_dict:
            return []

        spans = []

        def traverse(span: Dict):
            if span:
                spans.append(span)
                for child in span.get("children", []):
                    traverse(child)

        traverse(trace_dict)
        return spans

    def _is_read_only(self, tool_name: str) -> bool:
        """A tool is cacheable when no allowlist was supplied (default: treat
        every tool as read-only) or when it appears in the allowlist."""
        if self.read_only_tools is None:
            return True
        return tool_name in self.read_only_tools

    @staticmethod
    def _args_signature(span: Dict) -> Tuple:
        """Normalize a span's input into a hashable signature.

        Mirrors ``AgentLoopDetectionMetric._score_tool_repetition`` so the two
        metrics agree on what "the same call" means: JSON strings are parsed
        when possible and dict keys are sorted so argument order never
        affects identity.
        """
        input_val = span.get("input", {})
        if isinstance(input_val, str):
            try:
                input_val = json.loads(input_val)
            except Exception:
                pass

        if isinstance(input_val, dict):
            return tuple(sorted((str(k), str(v)) for k, v in input_val.items()))
        return (str(input_val),)

    def _score_duplicate_calls(
        self, tool_spans: List[Dict]
    ) -> Tuple[float, str, List[Tuple[str, int]]]:
        """Score the fraction of read-only tool calls that were not repeats.

        A call is redundant when an identical ``(name, args)`` read-only call
        already appeared in the trace. Unlike a loop alarm this fires on the
        very first repeat -- two identical cacheable calls are already waste.
        """
        read_only_spans = [
            s for s in tool_spans if self._is_read_only(s.get("name", ""))
        ]
        total = len(read_only_spans)
        if total <= 1:
            return 1.0, "Not enough read-only tool calls to be redundant.", []

        counts: Dict[Tuple[str, Tuple], int] = {}
        for span in read_only_spans:
            key = (span.get("name", ""), self._args_signature(span))
            counts[key] = counts.get(key, 0) + 1

        redundant = [(k[0], c) for k, c in counts.items() if c > 1]
        redundant_calls = sum(c - 1 for _, c in redundant)

        if redundant_calls == 0:
            return 1.0, "No redundant tool calls.", []

        score = (total - redundant_calls) / total
        offenders = ", ".join(
            f"'{name}' called {count} times with identical arguments"
            for name, count in sorted(redundant)
        )
        return (
            score,
            f"{redundant_calls} of {total} read-only tool call(s) were "
            f"redundant re-fetches: {offenders}.",
            sorted(redundant),
        )

    @staticmethod
    def _flatten_to_text(value: Any) -> str:
        """Flatten an arbitrary span input/output into searchable text.

        ``BaseSpan.input``/``output`` are typed ``Optional[Any]``: an LLM span
        input is typically a list of message dicts, a tool input a dict, and a
        tool output often a plain string. Everything is reduced to one
        lowercased, whitespace-collapsed string so the consumption check never
        depends on a particular shape (and never raises on a non-string).

        Structural JSON punctuation (braces, brackets, quotes, escapes) is
        stripped so that the *content* is compared rather than its encoding.
        Without this, a dict tool output serialized into an LLM message would
        carry escaped quotes (``{\\"policy\\": ...}``) and would never match
        the same payload serialized on its own -- reporting a genuinely
        consumed output as unconsumed.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, sort_keys=True, default=str)
            except (TypeError, ValueError):
                text = str(value)
        text = text.lower()
        for ch in ('\\"', "\\", '"', "'", "{", "}", "[", "]", ",", ":"):
            text = text.replace(ch, " ")
        return " ".join(text.split())

    def _score_unconsumed_output(
        self, trace_dict: Dict
    ) -> Tuple[float, str, List[str]]:
        """Score the fraction of tool outputs that reached an LLM.

        A tool output is *consumed* when its normalized text appears in the
        normalized input of some LLM span that is not one of its ancestors.
        Ancestors are excluded because a parent span's input was fixed before
        its child tool ever ran, so it cannot have consumed the output.
        """
        # Collect every tool span together with the identity of its ancestors,
        # plus every LLM span's input text keyed by span identity.
        tool_entries: List[Tuple[int, Dict, Set[int]]] = []
        llm_inputs: List[Tuple[int, str]] = []

        def traverse(span: Dict, ancestors: Tuple[int, ...]):
            span_id = id(span)
            span_type = span.get("type")
            if span_type == "tool":
                tool_entries.append((span_id, span, set(ancestors)))
            elif span_type == "llm":
                llm_inputs.append(
                    (span_id, self._flatten_to_text(span.get("input")))
                )
            for child in span.get("children", []):
                traverse(child, ancestors + (span_id,))

        traverse(trace_dict, ())

        if not tool_entries:
            return 1.0, "No tool spans found.", []

        considered = 0
        unconsumed: List[str] = []
        for tool_id, span, ancestor_ids in tool_entries:
            needle = self._flatten_to_text(span.get("output"))
            # Outputs too short to match distinctively are not judged -- a
            # two-character output would collide with unrelated prose.
            if len(needle) < _MIN_CONSUMPTION_MATCH_LENGTH:
                continue

            considered += 1
            consumed = any(
                llm_id not in ancestor_ids
                and llm_id != tool_id
                and needle in haystack
                for llm_id, haystack in llm_inputs
            )
            if not consumed:
                unconsumed.append(span.get("name", "unnamed"))

        if considered == 0:
            return (
                1.0,
                "No tool outputs long enough to check for consumption.",
                [],
            )

        if not unconsumed:
            return 1.0, "All tool outputs were consumed.", []

        score = (considered - len(unconsumed)) / considered
        names = ", ".join(f"'{n}'" for n in sorted(set(unconsumed)))
        return (
            score,
            f"{len(unconsumed)} of {considered} tool output(s) were never "
            f"passed to an LLM span: {names}.",
            sorted(set(unconsumed)),
        )

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Redundant Tool Call"
