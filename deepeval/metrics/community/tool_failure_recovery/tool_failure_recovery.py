import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.tool_failure_recovery.template import (
    ToolFailureRecoveryTemplate,
)
from deepeval.metrics.community.tool_failure_recovery.schema import (
    HallucinatedSuccessVerdicts,
    RecoveryVerdicts,
)

# Patterns that mark a string tool output as a failure when the span carries
# no explicit ``error`` field. Anchored at the start of the output so that
# ordinary results which merely *mention* errors ("No errors found") are not
# false-positived. Overridable via the ``error_patterns`` constructor arg.
_DEFAULT_ERROR_PATTERNS = [
    r"^\s*error\b",
    r"^\s*exception\b",
    r"^\s*traceback\b",
    r"^\s*fatal\b",
    r"^\s*failed\b",
    r"^\s*failure\b",
    r"^\s*timeout\b",
    r"^\s*timed\s+out\b",
    r"^\s*\[?\s*\w+(error|exception)\s*\]?\s*:",
]

# Max characters of any single input/output/error shown to the judge.
_JUDGE_FIELD_LIMIT = 800


class ToolFailureRecoveryMetric(BaseMetric):
    """Measures how an LLM agent behaves after a tool call fails mid-trace.

    Reads the agent's execution trace (``@observe`` tracing) and evaluates
    three sub-signals over every failed tool call (error result, raised
    exception, timeout):

    1. **Retry discipline** (deterministic, no LLM) — after a failure, blind
       *identical* retries (same tool name + same arguments after shape
       normalization) beyond ``max_blind_retries`` are penalized. Adjusted
       retries (changed arguments) are never penalized.
    2. **Hallucinated success** (LLM-judged) — downstream reasoning or the
       final answer claims/implies results from a call that actually failed.
       This is the deadliest failure mode and caps the score at 0.
    3. **Recovery quality** (LLM-judged) — did the agent adapt (alternative
       tool, re-plan, adjusted retry) or honestly degrade ("couldn't complete
       X, here's what I have"), versus silently dropping the sub-task?

    Score formula::

        if no failed tool calls:      score = 1.0
        elif any hallucinated success: score = 0.0
        else: score = min(retry_discipline_score, recovery_quality_score)

    A trace with no failed tool calls scores ``1.0`` (nothing to recover
    from), mirroring how ``AgentLoopDetectionMetric`` scores clean traces —
    and no LLM call is made in that case.

    Failure detection is heuristic where tracing does not record an explicit
    error: a tool span counts as *failed* when (a) its ``error`` field is
    set, (b) its dict output carries a truthy ``error`` key or an
    error/failed/timeout ``status``, or (c) its string output matches one of
    ``error_patterns`` (anchored regexes, see ``_DEFAULT_ERROR_PATTERNS``).
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        max_blind_retries: int = 1,
        error_patterns: Optional[List[str]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.max_blind_retries = max_blind_retries
        self.error_patterns = (
            error_patterns
            if error_patterns is not None
            else list(_DEFAULT_ERROR_PATTERNS)
        )
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
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
            False,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                analysis = self._analyze_trace(test_case)
                if analysis is not None and analysis["failed_calls"]:
                    prompts = self._build_judge_prompts(analysis, test_case)
                    hallucination_verdicts = (
                        self._generate_hallucination_verdicts(prompts[0])
                    )
                    recovery_verdicts = self._generate_recovery_verdicts(
                        prompts[1]
                    )
                else:
                    hallucination_verdicts = None
                    recovery_verdicts = None
                self._finalize(
                    analysis, hallucination_verdicts, recovery_verdicts
                )

            return self.score

    async def a_measure(
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
            False,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            analysis = self._analyze_trace(test_case)
            if analysis is not None and analysis["failed_calls"]:
                prompts = self._build_judge_prompts(analysis, test_case)
                (
                    hallucination_verdicts,
                    recovery_verdicts,
                ) = await asyncio.gather(
                    self._a_generate_hallucination_verdicts(prompts[0]),
                    self._a_generate_recovery_verdicts(prompts[1]),
                )
            else:
                hallucination_verdicts = None
                recovery_verdicts = None
            self._finalize(analysis, hallucination_verdicts, recovery_verdicts)
            return self.score

    ###################################
    # Deterministic trace analysis
    ###################################

    def _analyze_trace(self, test_case: LLMTestCase) -> Optional[Dict]:
        """Extract tool spans, detect failures, and score retry discipline.

        Returns ``None`` when the test case carries no trace, or a dict:
        ``all_spans``, ``tool_spans``, ``failed_calls`` (list of dicts with
        ``failure_index``, ``span_index``, ``name``, ``args``, ``error``),
        ``retry_score``, ``retry_details``, ``blind_retry_total``.
        """
        if test_case._trace_dict is None:
            return None

        all_spans = self._extract_all_spans(test_case._trace_dict)
        tool_spans = [s for s in all_spans if s.get("type") == "tool"]
        failed_flags = [self._is_failed_tool_span(s) for s in tool_spans]

        failed_calls = []
        for index, (span, failed) in enumerate(zip(tool_spans, failed_flags)):
            if failed:
                failed_calls.append(
                    {
                        "failure_index": len(failed_calls) + 1,
                        "span_index": index,
                        "name": span.get("name", ""),
                        "args": self._display_value(span.get("input", {})),
                        "error": self._display_value(
                            span.get("error") or span.get("output", "")
                        ),
                    }
                )

        retry_score, retry_details, blind_retry_total = (
            self._score_retry_discipline(tool_spans, failed_flags)
        )

        return {
            "all_spans": all_spans,
            "tool_spans": tool_spans,
            "failed_flags": failed_flags,
            "failed_calls": failed_calls,
            "retry_score": retry_score,
            "retry_details": retry_details,
            "blind_retry_total": blind_retry_total,
        }

    def _extract_all_spans(self, trace_dict: Optional[Dict]) -> List[Dict]:
        """Pre-order traversal of the nested ``children`` tree — the same
        chronological approximation ``AgentLoopDetectionMetric`` uses."""
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

    def _is_failed_tool_span(self, span: Dict) -> bool:
        error = span.get("error")
        if isinstance(error, str):
            if error.strip():
                return True
        elif error is not None:
            return True

        output = span.get("output")
        if isinstance(output, dict):
            if output.get("error"):
                return True
            status = str(output.get("status", "")).strip().lower()
            if status in ("error", "failed", "failure", "timeout"):
                return True
        elif isinstance(output, str):
            for pattern in self.error_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    return True
        return False

    @staticmethod
    def _normalize_shape(value: Any) -> Any:
        """Recursively normalize an args structure so that cosmetically
        different but semantically identical arguments compare equal:
        dict keys stringified (ordering handled by sorted-key dumps), and
        strings whitespace-collapsed."""
        if isinstance(value, dict):
            return {
                str(k): ToolFailureRecoveryMetric._normalize_shape(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                ToolFailureRecoveryMetric._normalize_shape(v) for v in value
            ]
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @classmethod
    def _call_signature(cls, span: Dict) -> Tuple[str, str]:
        """``(tool_name, canonical_json_args)`` identity for retry matching.

        String inputs are parsed as JSON when possible (tool inputs are
        frequently serialized dicts — same treatment as
        ``AgentLoopDetectionMetric._score_tool_repetition``), then the
        structure is shape-normalized and dumped with sorted keys.
        """
        name = span.get("name", "")
        input_val = span.get("input", {})
        if isinstance(input_val, str):
            try:
                input_val = json.loads(input_val)
            except Exception:
                pass
        normalized = cls._normalize_shape(input_val)
        try:
            canonical = json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            canonical = str(normalized)
        return (name, canonical)

    def _score_retry_discipline(
        self, tool_spans: List[Dict], failed_flags: List[bool]
    ) -> Tuple[float, List[str], int]:
        """Score blind identical retries after each failure.

        For each failed call, count the run of *immediately consecutive*
        tool calls with an identical ``(name, canonical_args)`` signature.
        A different tool or changed arguments breaks the run — that is
        adaptation, never penalized. Per failure:

        - ``retries <= max_blind_retries``      -> 1.0 (a bounded retry is
          reasonable; transient errors exist)
        - ``retries == max_blind_retries + 1``  -> 0.5
        - ``retries >= max_blind_retries + 2``  -> 0.0 (retry storm)

        The sub-signal score is the **minimum** across failures (worst
        offender), mirroring how a single severe signal drives
        ``AgentLoopDetectionMetric`` sub-scores to 0.
        """
        worst = 1.0
        details: List[str] = []
        blind_retry_total = 0
        consumed = set()

        for index, failed in enumerate(failed_flags):
            if not failed or index in consumed:
                continue
            signature = self._call_signature(tool_spans[index])
            retries = 0
            next_index = index + 1
            while (
                next_index < len(tool_spans)
                and self._call_signature(tool_spans[next_index]) == signature
            ):
                retries += 1
                consumed.add(next_index)
                next_index += 1

            blind_retry_total += retries
            if retries <= self.max_blind_retries:
                call_score = 1.0
            elif retries == self.max_blind_retries + 1:
                call_score = 0.5
            else:
                call_score = 0.0

            if call_score < 1.0:
                details.append(
                    f"Tool '{tool_spans[index].get('name', '')}' was blindly "
                    f"retried {retries} time(s) with identical arguments "
                    f"after failing (allowance: {self.max_blind_retries})."
                )
            worst = min(worst, call_score)

        if not details:
            details.append("No blind identical retries beyond the allowance.")
        return worst, details, blind_retry_total

    ###################################
    # Judge prompt construction
    ###################################

    @staticmethod
    def _display_value(value: Any, limit: int = _JUDGE_FIELD_LIMIT) -> str:
        if isinstance(value, (dict, list)):
            try:
                text = json.dumps(value, default=str)
            except (TypeError, ValueError):
                text = str(value)
        else:
            text = str(value)
        if len(text) > limit:
            text = text[:limit] + "... [truncated]"
        return text

    def _render_trace_summary(self, analysis: Dict) -> str:
        """Linearize the trace for the judge: every span in pre-order, with
        failed tool calls tagged ``FAILURE #N`` so verdicts can be joined
        back by ``failure_index``."""
        failure_by_span_id = {
            id(analysis["tool_spans"][fc["span_index"]]): fc["failure_index"]
            for fc in analysis["failed_calls"]
        }

        lines = []
        step = 0
        for span in analysis["all_spans"]:
            step += 1
            span_type = span.get("type", "unknown")
            name = span.get("name", "unnamed")
            if span_type == "tool":
                args = self._display_value(span.get("input", {}))
                failure_index = failure_by_span_id.get(id(span))
                if failure_index is not None:
                    error = self._display_value(
                        span.get("error") or span.get("output", "")
                    )
                    lines.append(
                        f"{step}. [TOOL - FAILURE #{failure_index}] "
                        f"{name}({args}) FAILED with: {error}"
                    )
                else:
                    output = self._display_value(span.get("output", ""))
                    lines.append(
                        f"{step}. [TOOL] {name}({args}) returned: {output}"
                    )
            else:
                output = self._display_value(span.get("output", ""))
                lines.append(
                    f"{step}. [{span_type.upper()}] {name} output: {output}"
                )
        return "\n".join(lines)

    def _render_failed_calls_block(self, analysis: Dict) -> str:
        lines = []
        for fc in analysis["failed_calls"]:
            lines.append(
                f"FAILURE #{fc['failure_index']}: {fc['name']}"
                f"({fc['args']}) failed with: {fc['error']}"
            )
        return "\n".join(lines)

    def _build_judge_prompts(
        self, analysis: Dict, test_case: LLMTestCase
    ) -> Tuple[str, str]:
        trace_summary = self._render_trace_summary(analysis)
        failed_calls_block = self._render_failed_calls_block(analysis)
        final_output = str(test_case.actual_output)
        hallucination_prompt = (
            ToolFailureRecoveryTemplate.generate_hallucinated_success_verdicts(
                trace_summary=trace_summary,
                failed_calls_block=failed_calls_block,
                final_output=final_output,
            )
        )
        recovery_prompt = (
            ToolFailureRecoveryTemplate.generate_recovery_verdicts(
                trace_summary=trace_summary,
                failed_calls_block=failed_calls_block,
                final_output=final_output,
            )
        )
        return hallucination_prompt, recovery_prompt

    ###################################
    # Judge calls
    ###################################

    def _generate_hallucination_verdicts(
        self, prompt: str
    ) -> HallucinatedSuccessVerdicts:
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=HallucinatedSuccessVerdicts,
            extract_schema=lambda s: s,
            extract_json=lambda data: HallucinatedSuccessVerdicts(**data),
        )

    async def _a_generate_hallucination_verdicts(
        self, prompt: str
    ) -> HallucinatedSuccessVerdicts:
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=HallucinatedSuccessVerdicts,
            extract_schema=lambda s: s,
            extract_json=lambda data: HallucinatedSuccessVerdicts(**data),
        )

    def _generate_recovery_verdicts(self, prompt: str) -> RecoveryVerdicts:
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=RecoveryVerdicts,
            extract_schema=lambda s: s,
            extract_json=lambda data: RecoveryVerdicts(**data),
        )

    async def _a_generate_recovery_verdicts(
        self, prompt: str
    ) -> RecoveryVerdicts:
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=RecoveryVerdicts,
            extract_schema=lambda s: s,
            extract_json=lambda data: RecoveryVerdicts(**data),
        )

    ###################################
    # Scoring
    ###################################

    _RECOVERY_VALUES = {"recovered": 1.0, "partial": 0.5, "ignored": 0.0}

    def _finalize(
        self,
        analysis: Optional[Dict],
        hallucination_verdicts: Optional[HallucinatedSuccessVerdicts],
        recovery_verdicts: Optional[RecoveryVerdicts],
    ):
        """Combine sub-signals into the final score, reason, and logs."""
        if analysis is None:
            self.score = 0.0
            self.success = False
            self.reason = (
                "No trace data found. This metric requires trace "
                "data from @observe."
            )
            self.verbose_logs = ""
            return

        failed_calls = analysis["failed_calls"]

        if not failed_calls:
            self.score_breakdown = {
                "retry_discipline": 1.0,
                "hallucinated_success": 1.0,
                "recovery_quality": 1.0,
            }
            self.score = 1.0
            self.success = self.is_successful()
            self.reason = (
                "No failed tool calls were observed in the trace, so there "
                "was nothing to recover from."
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Tool spans inspected: {len(analysis['tool_spans'])}",
                    "Failed tool calls: 0",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return

        retry_score = analysis["retry_score"]

        hallucinated = [
            v
            for v in (
                hallucination_verdicts.verdicts
                if hallucination_verdicts
                else []
            )
            if v.verdict.strip().lower() == "hallucinated"
        ]
        hallucination_score = 0.0 if hallucinated else 1.0

        recovery_values = [
            self._RECOVERY_VALUES.get(v.verdict.strip().lower(), 0.0)
            for v in (recovery_verdicts.verdicts if recovery_verdicts else [])
        ]
        recovery_score = (
            sum(recovery_values) / len(recovery_values)
            if recovery_values
            else 1.0
        )

        self.score_breakdown = {
            "retry_discipline": retry_score,
            "hallucinated_success": hallucination_score,
            "recovery_quality": recovery_score,
        }
        # score = min(retry_discipline, recovery_quality), capped at 0 when
        # any failed call's results were hallucinated as successful — a
        # fabricated tool result must never pass.
        if hallucinated:
            self.score = 0.0
        else:
            self.score = min(retry_score, recovery_score)
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.success = self.is_successful()
        self.reason = self._generate_reason(
            analysis, hallucinated, hallucination_verdicts, recovery_verdicts
        )
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Failed tool calls ({len(failed_calls)}):\n"
                + self._render_failed_calls_block(analysis),
                f"Retry Discipline Score: {retry_score} "
                f"({' '.join(analysis['retry_details'])})",
                f"Hallucinated Success Score: {hallucination_score} "
                f"({len(hallucinated)} hallucinated verdict(s))",
                f"Recovery Quality Score: {recovery_score}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    def _generate_reason(
        self,
        analysis: Dict,
        hallucinated: List,
        hallucination_verdicts: Optional[HallucinatedSuccessVerdicts],
        recovery_verdicts: Optional[RecoveryVerdicts],
    ) -> Optional[str]:
        if self.include_reason is False:
            return None

        failed_calls = analysis["failed_calls"]
        failed_names = ", ".join(f"'{fc['name']}'" for fc in failed_calls)
        parts = [
            f"Observed {len(failed_calls)} failed tool call(s): "
            f"{failed_names}."
        ]

        parts.append(
            f"Blind identical retries detected: "
            f"{analysis['blind_retry_total']} "
            f"(allowance per failure: {self.max_blind_retries})."
        )
        if analysis["retry_score"] < 1.0:
            parts.extend(analysis["retry_details"])

        if hallucinated:
            worst = hallucinated[0]
            detail = f" {worst.reasoning}" if worst.reasoning else ""
            parts.append(
                f"Hallucinated success on {len(hallucinated)} failed "
                f"call(s) — the agent presented results from a failed tool "
                f"call as real, which caps the score at 0.{detail}"
            )
        elif hallucination_verdicts and hallucination_verdicts.verdicts:
            parts.append(
                "No hallucinated success: downstream output never claims "
                "results from a failed call."
            )

        if recovery_verdicts and recovery_verdicts.verdicts:
            counts: Dict[str, int] = {}
            worst_reasoning = None
            worst_value = 2.0
            for v in recovery_verdicts.verdicts:
                verdict = v.verdict.strip().lower()
                counts[verdict] = counts.get(verdict, 0) + 1
                value = self._RECOVERY_VALUES.get(verdict, 0.0)
                if value < worst_value:
                    worst_value = value
                    worst_reasoning = v.reasoning
            summary = ", ".join(
                f"{count} {verdict}" for verdict, count in counts.items()
            )
            recovery_part = f"Recovery verdicts: {summary}."
            if worst_value < 1.0 and worst_reasoning:
                recovery_part += f" {worst_reasoning}"
            parts.append(recovery_part)

        return " ".join(parts)

    @property
    def __name__(self):
        return "Tool Failure Recovery"
