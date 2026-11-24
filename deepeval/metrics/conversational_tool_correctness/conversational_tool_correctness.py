"""Tool Correctness for conversations: every tool the assistant called across
the conversation is compared with the test case's ``expected_tools``."""

from typing import List, Optional

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.tool_correctness.tool_correctness import (
    ToolCorrectnessMetric,
)
from deepeval.metrics.utils import construct_verbose_logs, prepare_measure
from deepeval.test_case import (
    ConversationalTestCase,
    MultiTurnParams,
    ToolCall,
    ToolCallParams,
    Turn,
)


class _ToolCallScorer(ToolCorrectnessMetric):
    """ToolCorrectnessMetric's deterministic matching, without a judge model.

    Skips the parent ``__init__`` (which sets up an LLM for tool selection)
    and only sets what the matching and reason helpers read.
    """

    def __init__(
        self,
        evaluation_params: List[ToolCallParams],
        should_exact_match: bool,
        should_consider_ordering: bool,
    ):
        self.evaluation_params = evaluation_params
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering
        self.strict_mode = False
        self.threshold = 0

    def score(
        self, tools_called: List[ToolCall], expected_tools: List[ToolCall]
    ) -> "tuple[float, str]":
        self.tools_called = tools_called
        self.expected_tools = expected_tools
        return self._calculate_score(), self._generate_reason()


class ConversationalToolCorrectnessMetric(BaseConversationalMetric):
    """Collects the tools the assistant called across every turn, in order,
    and scores them against the test case's ``expected_tools`` with the Tool
    Correctness rules. No LLM is used."""

    _required_test_case_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.EXPECTED_TOOLS,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_params: Optional[List[ToolCallParams]] = None,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_params = list(evaluation_params or [])
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.model = None
        self.using_native_model = None
        self.evaluation_model = None

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            self.tools_called = self._tools_called(test_case.turns)
            self.expected_tools = test_case.expected_tools
            scorer = _ToolCallScorer(
                self.evaluation_params,
                self.should_exact_match,
                self.should_consider_ordering,
            )
            score, reason = scorer.score(self.tools_called, self.expected_tools)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.reason = reason if self.include_reason else None
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Tools called across the conversation: {[t.name for t in self.tools_called]}",
                    f"Expected tools: {[t.name for t in self.expected_tools]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @staticmethod
    def _tools_called(turns: List[Turn]) -> List[ToolCall]:
        return [
            tool
            for turn in turns
            if turn.role == "assistant"
            for tool in (turn.tools_called or [])
        ]

    @property
    def __name__(self):
        return "Conversational Tool Correctness"
