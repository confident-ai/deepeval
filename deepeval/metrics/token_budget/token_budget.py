from typing import Optional, List, Dict, Tuple

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.metrics.indicator import metric_progress_indicator


class TokenBudgetMetric(BaseMetric):
    """Gates an agent run against token and/or cost budgets.

    Walks the agent's execution trace, sums token usage and cost across every
    LLM span, and checks the totals against the budgets you set.  This is a
    fully **deterministic** metric — no LLM, no API key — so it is cheap and
    reliable to run as a CI gate or in production to catch runs that blow past
    their expected token / cost envelope.

    You enable a budget simply by setting it (any budget left ``None`` is
    disabled):

    - ``max_total_tokens`` — cap on ``input + output`` tokens across all LLM spans.
    - ``max_input_tokens`` — cap on input (prompt) tokens only.
    - ``max_output_tokens`` — cap on output (completion) tokens only.
    - ``max_cost`` — cap on total dollar cost across all LLM spans.

    At least one budget must be provided.

    Scoring
    ~~~~~~~
    The score is a **graded ratio**, not a simple pass/fail.  For each enabled
    budget the metric computes ``spent / limit``:

    - If the run is under the budget, that budget's sub-score is ``1.0``.
    - If it is over, the sub-score degrades as ``limit / spent`` — so being
      twice over budget scores ``0.5``, three times over scores ``0.33``, etc.

    The overall score is the **minimum** across all enabled budgets: the run is
    only as good as its worst-breached budget.  This makes the metric a strict
    gate — a single breached budget fails the run.  Under ``strict_mode`` the
    score collapses to a binary ``1.0`` (all budgets respected) or ``0.0`` (any
    budget breached).

    Cost accounting
    ~~~~~~~~~~~~~~~
    The trace dict exposes ``cost_per_input_token`` and ``cost_per_output_token``
    as *per-token rates* (not totals), so a span's cost is computed as::

        input_token_count  * cost_per_input_token
      + output_token_count * cost_per_output_token

    A span only contributes to the cost total when both the token count *and*
    its per-token rate are present.  Spans that report tokens but no per-token
    rate are counted as ``unpriced`` and surfaced in the reason so a
    cheap-looking run is never silently trusted.

    Design decisions
    ~~~~~~~~~~~~~~~~
    * **Fully deterministic** — no ``model`` parameter.  Every value is read
      straight from the trace and summed; an LLM would add nothing.  This keeps
      the metric zero-cost and safe to run on every request in production.
    * **Graded, not binary (by default)** — a graded ratio tells you *how far*
      over budget a run went, which is far more actionable than a bare
      pass/fail.  ``strict_mode`` is available when a binary gate is preferred.
    * **Missing token data is reported, not guessed** — spans without token
      counts contribute ``0`` but are counted separately and reported, so the
      score is never silently based on partial data.

    Limitations
    ~~~~~~~~~~~
    * **No latency / duration budget.**  ``create_nested_spans_dict`` strips
      ``start_time`` and ``end_time`` (along with ``uuid`` and ``status``) from
      the trace dict, so span timing is simply unavailable to this metric.  It
      can gate tokens and cost but *cannot* gate wall-clock latency.
    * **Cost requires per-token rates.**  Integrations that record token counts
      but not ``cost_per_*_token`` will produce a token total but no cost for
      those spans.  The ``max_cost`` budget is only meaningful when your
      integration populates per-token pricing.
    * **Token counts are provider-reported** and are only as accurate as the
      integration that recorded them.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        max_total_tokens: Optional[float] = None,
        max_input_tokens: Optional[float] = None,
        max_output_tokens: Optional[float] = None,
        max_cost: Optional[float] = None,
        threshold: float = 1.0,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        if (
            max_total_tokens is None
            and max_input_tokens is None
            and max_output_tokens is None
            and max_cost is None
        ):
            raise ValueError(
                "TokenBudgetMetric requires at least one budget: set one or "
                "more of `max_total_tokens`, `max_input_tokens`, "
                "`max_output_tokens`, or `max_cost`."
            )

        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_cost = max_cost

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
        _log_metric_to_confident: bool = True,
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
            self._calculate_metric(test_case)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
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

        all_spans = self._extract_all_spans(test_case._trace_dict)
        llm_spans = [s for s in all_spans if s.get("type") == "llm"]

        usage = self._aggregate_usage(llm_spans)

        sub_scores: Dict[str, float] = {}
        breakdown: Dict[str, Dict] = {}
        breach_reasons: List[str] = []

        budgets = [
            ("total_tokens", self.max_total_tokens, usage["total_tokens"]),
            ("input_tokens", self.max_input_tokens, usage["input_tokens"]),
            ("output_tokens", self.max_output_tokens, usage["output_tokens"]),
            ("cost", self.max_cost, usage["cost"]),
        ]

        for name, limit, spent in budgets:
            if limit is None:
                continue
            sub_score, ratio = self._budget_sub_score(spent, limit)
            sub_scores[name] = sub_score
            breakdown[name] = {
                "spent": spent,
                "limit": limit,
                "ratio": ratio,
                "sub_score": sub_score,
            }
            if sub_score < 1.0:
                breach_reasons.append(
                    self._format_breach(name, spent, limit, ratio)
                )

        # sub_scores is guaranteed non-empty: __init__ requires >= 1 budget.
        raw_score = min(sub_scores.values())

        if self.strict_mode:
            self.score = 1.0 if raw_score >= 1.0 else 0.0
        else:
            self.score = raw_score

        breakdown["most_expensive_span"] = usage["most_expensive_span"]
        breakdown["llm_span_count"] = usage["llm_span_count"]
        breakdown["untokened_spans"] = usage["untokened_spans"]
        breakdown["unpriced_spans"] = usage["unpriced_spans"]
        self.score_breakdown = breakdown

        self.success = self.score >= self.threshold
        self.reason = self._build_reason(usage, breach_reasons)

        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"LLM spans: {usage['llm_span_count']} "
                f"(untokened: {usage['untokened_spans']}, "
                f"unpriced: {usage['unpriced_spans']})",
                f"Total tokens: {usage['total_tokens']} "
                f"(input: {usage['input_tokens']}, "
                f"output: {usage['output_tokens']})",
                f"Total cost: {usage['cost']}",
                f"Budget sub-scores: {sub_scores}",
                f"Score: {self.score}",
                f"Reason: {self.reason}",
            ],
        )

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

    def _aggregate_usage(self, llm_spans: List[Dict]) -> Dict:
        """Sum tokens and cost across LLM spans, tracking data gaps.

        Token counts and per-token rates are ``Optional[float]`` on the trace
        span, so every access is null-guarded.  A span contributes to the cost
        total only when both its token count and the matching per-token rate
        are present; spans missing token data or pricing are counted so the
        reason can flag partial coverage.
        """
        total_input = 0.0
        total_output = 0.0
        total_cost = 0.0
        untokened_spans = 0
        unpriced_spans = 0
        most_expensive_span: Optional[Dict] = None

        for span in llm_spans:
            in_count = span.get("input_token_count")
            out_count = span.get("output_token_count")
            in_rate = span.get("cost_per_input_token")
            out_rate = span.get("cost_per_output_token")

            if in_count is None and out_count is None:
                untokened_spans += 1

            if in_count is not None:
                total_input += in_count
            if out_count is not None:
                total_output += out_count

            span_cost = 0.0
            span_priced = False
            if in_count is not None and in_rate is not None:
                span_cost += in_count * in_rate
                span_priced = True
            if out_count is not None and out_rate is not None:
                span_cost += out_count * out_rate
                span_priced = True

            has_tokens = in_count is not None or out_count is not None
            if has_tokens and not span_priced:
                unpriced_spans += 1

            if span_priced:
                total_cost += span_cost
                if (
                    most_expensive_span is None
                    or span_cost > most_expensive_span["cost"]
                ):
                    most_expensive_span = {
                        "name": span.get("name", "unnamed"),
                        "cost": span_cost,
                    }

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "cost": total_cost,
            "llm_span_count": len(llm_spans),
            "untokened_spans": untokened_spans,
            "unpriced_spans": unpriced_spans,
            "most_expensive_span": most_expensive_span,
        }

    @staticmethod
    def _budget_sub_score(
        spent: float, limit: float
    ) -> Tuple[float, Optional[float]]:
        """Return ``(sub_score, ratio)`` for one budget.

        ``ratio`` is ``spent / limit``.  A run under budget scores ``1.0``; a
        run over budget degrades as ``limit / spent`` (2x over -> 0.5).  A
        ``limit`` of ``0`` is a degenerate budget: any spend breaches it (0.0),
        no spend respects it (1.0).
        """
        if limit <= 0:
            if spent <= 0:
                return 1.0, 0.0 if limit == 0 else None
            return 0.0, None
        ratio = spent / limit
        if ratio <= 1.0:
            return 1.0, ratio
        return limit / spent, ratio

    @staticmethod
    def _format_breach(
        name: str, spent: float, limit: float, ratio: Optional[float]
    ) -> str:
        label = name.replace("_", " ")
        if name == "cost":
            spent_str = f"${spent:.6f}"
            limit_str = f"${limit:.6f}"
        else:
            spent_str = f"{spent:g}"
            limit_str = f"{limit:g}"
        over = f" ({ratio:.2f}x budget)" if ratio is not None else ""
        return f"{label} {spent_str} exceeded budget {limit_str}{over}."

    def _build_reason(self, usage: Dict, breach_reasons: List[str]) -> str:
        if not self.include_reason:
            return ""

        parts: List[str] = []
        if breach_reasons:
            parts.append(" ".join(breach_reasons))
        elif usage["llm_span_count"] == 0:
            # No LLM spans at all — the budget could not be measured, so we
            # must not claim it was "respected". The score stays 1.0 (an agent
            # legitimately may make no LLM calls) but the reason says so plainly.
            parts.append(
                "No LLM spans found in the trace; token/cost budgets could "
                "not be measured."
            )
        else:
            parts.append("All token and cost budgets respected.")

        if usage["untokened_spans"]:
            parts.append(
                f"{usage['untokened_spans']} of {usage['llm_span_count']} "
                f"LLM spans had no token data."
            )
        if usage["unpriced_spans"]:
            parts.append(
                f"{usage['unpriced_spans']} of {usage['llm_span_count']} "
                f"LLM spans had tokens but no per-token pricing "
                f"(excluded from cost)."
            )
        return " ".join(parts)

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
        return "Token Budget"
