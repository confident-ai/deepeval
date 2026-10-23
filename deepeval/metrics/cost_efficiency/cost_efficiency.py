from typing import Optional, List, Dict, Tuple

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.indicator import metric_progress_indicator


class CostEfficiencyMetric(BaseMetric):
    """Measures the token cost of an agent execution against a budget.

    Sums the ``input_token_count`` and ``output_token_count`` of every LLM
    span in a trace and scores how efficiently the agent used its token
    budget. Returns ``1.0`` when total tokens are within (or equal to) the
    budget, and degrades proportionally (``budget / total_tokens``) when the
    budget is exceeded.

    Design decisions
    ~~~~~~~~~~~~~~~~
    * **Fully deterministic** — no LLM / API key required. The metric is
      meant to run in production pipelines at zero cost and zero latency,
      so it only does arithmetic over token counts already recorded on the
      trace.
    * **No ``model`` parameter** — the score is a pure function of token
      counts; accepting a ``model`` argument would be misleading.

    Limitations
    ~~~~~~~~~~~
    * Token counts are only available when the LLM spans recorded them
      (``input_token_count`` / ``output_token_count``). If a model or
      integration does not populate these fields, the metric will report
      zero tokens and score ``1.0`` — check the ``score_breakdown`` to
      confirm the counts are meaningful.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        token_budget: float,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than 0.")

        self.token_budget = token_budget
        self.threshold = 1 if strict_mode else threshold
        self.model = None
        self.using_native_model = True
        self.evaluation_model = None
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
            test_case.multimodal,
        )

        self.evaluation_cost = 0

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
                return self.score
            else:
                self._calculate_metric(test_case)
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
            test_case.multimodal,
        )

        self.evaluation_cost = 0

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self._calculate_metric(test_case)
            return self.score

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

        total_input, total_output, llm_span_count = self._sum_llm_tokens(
            test_case._trace_dict
        )
        total_tokens = total_input + total_output

        if total_tokens == 0:
            score = 1.0
        else:
            score = min(1.0, self.token_budget / total_tokens)

        if self.strict_mode and score < self.threshold:
            score = 0.0

        self.score = score
        self.score_breakdown = {
            "total_tokens": total_tokens,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "token_budget": self.token_budget,
            "llm_span_count": llm_span_count,
        }
        self.success = self.is_successful()
        self.reason = self._build_reason(total_tokens)
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Total Tokens: {total_tokens:.0f} "
                f"(input={total_input:.0f}, output={total_output:.0f}, "
                f"llm_spans={llm_span_count})",
                f"Token Budget: {self.token_budget:.0f}",
                f"Score: {self.score:.2f}",
                f"Reason: {self.reason}",
            ],
        )

    def _sum_llm_tokens(self, trace_dict: Optional[Dict]) -> Tuple[float, float, int]:
        total_input = 0.0
        total_output = 0.0
        llm_span_count = 0

        for span in self._extract_all_spans(trace_dict):
            if span.get("type") != "llm":
                continue
            llm_span_count += 1
            total_input += span.get("input_token_count") or 0
            total_output += span.get("output_token_count") or 0

        return total_input, total_output, llm_span_count

    def _extract_all_spans(self, trace_dict: Optional[Dict]) -> List[Dict]:
        if not trace_dict:
            return []

        spans: List[Dict] = []

        def traverse(span: Dict):
            if span:
                spans.append(span)
                for child in span.get("children", []):
                    traverse(child)

        traverse(trace_dict)
        return spans

    def _build_reason(self, total_tokens: float) -> str:
        if total_tokens == 0:
            return (
                "No LLM token usage detected (0 tokens). Confirm the trace "
                "recorded token counts on its LLM spans."
            )
        if self.score >= 1.0:
            return (
                f"Total tokens ({total_tokens:.0f}) are within the "
                f"token budget ({self.token_budget:.0f})."
            )
        return (
            f"Total tokens ({total_tokens:.0f}) exceeded the token budget "
            f"({self.token_budget:.0f})."
        )

    @property
    def __name__(self):
        return "Cost Efficiency"
