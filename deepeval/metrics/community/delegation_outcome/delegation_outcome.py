from typing import Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

_MAX_ERROR_LOG_CHARS = 200


class DelegationOutcomeMetric(BaseMetric):
    """Did the sub-agents finish without errors?

    Applies the ``ToolOutcomeMetric`` approach to agent spans instead
    of tool spans: reads the agent spans of the trace captured with
    ``@observe`` and scores the fraction without a recorded ``error``.
    The root span is the delegation owner and is not scored — only
    nested (delegated) agents count. Scores ``1.0`` when no sub-agent
    errored, including when nothing was delegated. With
    ``ignore_recovered_failures=True``, an errored agent is forgiven
    when a later span for the same agent succeeded. Fully
    **deterministic**: no LLM, no API key, zero token cost.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1.0,
        ignore_recovered_failures: bool = False,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.ignore_recovered_failures = ignore_recovered_failures
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.requires_trace = True
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

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
            None,
            test_case.multimodal,
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if test_case._trace_dict is None:
                self.score = 0.0
                self.success = False
                self.reason = (
                    "No trace data found. This metric requires trace "
                    "data from @observe."
                )
                self.verbose_logs = ""
                return self.score

            agent_spans = [
                span
                for span in self._extract_child_spans(test_case._trace_dict)
                if str(span.get("type")).lower() == "agent"
            ]
            errored = [span for span in agent_spans if span.get("error")]
            counted = (
                self._unrecovered(agent_spans)
                if self.ignore_recovered_failures
                else errored
            )

            total = len(agent_spans)
            score = 1.0 if total == 0 else (total - len(counted)) / total
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(
                total, counted, len(errored) - len(counted)
            )
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Agents: {[span.get('name') for span in agent_spans]}",
                    "Errored agents:\n"
                    + (
                        "\n".join(
                            f"{span.get('name')}: "
                            f"{str(span.get('error'))[:_MAX_ERROR_LOG_CHARS]}"
                            for span in errored
                        )
                        or "None"
                    ),
                    f"Ignore recovered failures: "
                    f"{self.ignore_recovered_failures}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @staticmethod
    def _extract_child_spans(trace_dict: Dict) -> List[Dict]:
        """All spans except the root: the root owns the delegation."""
        spans: List[Dict] = []

        def traverse(span: Dict):
            if span:
                spans.append(span)
                for child in span.get("children", []):
                    traverse(child)

        for child in (trace_dict or {}).get("children", []):
            traverse(child)
        return spans

    @staticmethod
    def _unrecovered(agent_spans: List[Dict]) -> List[Dict]:
        unrecovered = []
        for index, span in enumerate(agent_spans):
            if not span.get("error"):
                continue
            recovered = any(
                later.get("name") == span.get("name") and not later.get("error")
                for later in agent_spans[index + 1 :]
            )
            if not recovered:
                unrecovered.append(span)
        return unrecovered

    def _generate_reason(
        self, total: int, counted: List[Dict], recovered_count: int
    ) -> Optional[str]:
        if not self.include_reason:
            return None
        if total == 0:
            return "No agents ran, so no delegation could fail."
        recovered_note = (
            f" {recovered_count} failed delegation(s) recovered and "
            "were not counted."
            if recovered_count
            else ""
        )
        if not counted:
            return (
                f"All {total} agent(s) completed without an "
                f"error.{recovered_note}"
            )
        counts: Dict[str, int] = {}
        for span in counted:
            name = span.get("name") or "unnamed"
            counts[name] = counts.get(name, 0) + 1
        breakdown = ", ".join(
            f"{name} ({count})" for name, count in counts.items()
        )
        return (
            f"{len(counted)} of {total} agent(s) errored: "
            f"{breakdown}.{recovered_note}"
        )

    @property
    def __name__(self):
        return "Delegation Outcome"
