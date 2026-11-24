"""Tool Correctness for conversations: each assistant turn's tool calls are
compared with that turn's ``expected_tools``."""

from typing import List, Optional, Tuple

from deepeval.errors import MissingTestCaseParamsError
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


class _TurnScorer(ToolCorrectnessMetric):
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

    def score_turn(
        self, tools_called: List[ToolCall], expected_tools: List[ToolCall]
    ) -> Tuple[float, str]:
        self.tools_called = tools_called
        self.expected_tools = expected_tools
        return self._calculate_score(), self._generate_reason()


class ConversationalToolCorrectnessMetric(BaseConversationalMetric):
    """Scores every assistant turn that sets ``expected_tools`` with the Tool
    Correctness rules, and averages the turn scores. Turns without
    ``expected_tools`` are not scored. No LLM is used."""

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
        scored_turns = self._scored_turns(test_case.turns)
        if not scored_turns:
            self.error = (
                "No assistant turn in the conversational test case sets "
                f"'expected_tools', which the '{self.__name__}' metric needs. "
                "Set Turn.expected_tools on the assistant turns to check "
                "([] means the turn should call no tools)."
            )
            raise MissingTestCaseParamsError(self.error)

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            scorer = _TurnScorer(
                self.evaluation_params,
                self.should_exact_match,
                self.should_consider_ordering,
            )
            self.turn_scores = []
            for index, turn in scored_turns:
                score, reason = scorer.score_turn(
                    turn.tools_called or [], turn.expected_tools
                )
                self.turn_scores.append(
                    {"turn": index, "score": score, "reason": reason}
                )

            score = sum(t["score"] for t in self.turn_scores) / len(
                self.turn_scores
            )
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.reason = self._reason() if self.include_reason else None
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    *(
                        f"Turn {t['turn']}: score {t['score']:.2f}. {t['reason']}"
                        for t in self.turn_scores
                    ),
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
    def _scored_turns(turns: List[Turn]) -> List[Tuple[int, Turn]]:
        return [
            (index, turn)
            for index, turn in enumerate(turns)
            if turn.role == "assistant" and turn.expected_tools is not None
        ]

    def _reason(self) -> str:
        total = len(self.turn_scores)
        failing = [t for t in self.turn_scores if t["score"] < 1]
        summary = (
            f"The score is {self.score:.2f}: {total - len(failing)} of "
            f"{total} scored assistant turns called the expected tools."
        )
        if not failing:
            return summary
        details = "\n".join(
            f"- Turn {t['turn']} ({t['score']:.2f}): {t['reason']}"
            for t in failing
        )
        return f"{summary}\n{details}"

    @property
    def __name__(self):
        return "Conversational Tool Correctness"
