from typing import Dict, List, Optional, Tuple

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


class _ToolCallScorer(ToolCorrectnessMetric):
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
    ) -> Tuple[float, str]:
        self.tools_called = tools_called
        self.expected_tools = expected_tools
        return self._calculate_score(), self._generate_reason()


class ConversationToolCorrectnessMetric(BaseConversationalMetric):
    _required_test_case_params = [
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
        MultiTurnParams.EXPECTED_TOOLS,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
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
        turns = test_case.turns
        pinned = self._pinned_tools(turns, test_case.expected_tools)
        unpinned = [t for t in test_case.expected_tools if t.turn is None]

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            checks = [
                (f"Turn {index}", turns[index].tools_called or [], tools)
                for index, tools in sorted(pinned.items())
            ]
            scorer = _ToolCallScorer(
                self.evaluation_params,
                self.should_exact_match,
                self.should_consider_ordering,
            )
            if unpinned or not pinned:
                label = "Any turn" if pinned else "Conversation"
                called = self._tools_left_for_any_turn(turns, pinned, scorer)
                checks.append((label, called, unpinned))

            self.check_scores = []
            for label, called, expected in checks:
                score, reason = scorer.score(called, expected)
                self.check_scores.append(
                    {"check": label, "score": score, "reason": reason}
                )
            self.tools_called = self._tools_called(turns)

            score = sum(c["score"] for c in self.check_scores) / len(checks)
            self.score = (
                0 if self.strict_mode and score < self.threshold else score
            )
            self.reason = self._reason() if self.include_reason else None
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    *(
                        f"{c['check']}: score {c['score']:.2f}. {c['reason']}"
                        for c in self.check_scores
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
    def _tools_called(turns: List[Turn]) -> List[ToolCall]:
        return [
            tool
            for turn in turns
            if turn.role == "assistant"
            for tool in (turn.tools_called or [])
        ]

    @staticmethod
    def _tools_left_for_any_turn(
        turns: List[Turn],
        pinned: Dict[int, List[ToolCall]],
        scorer: _ToolCallScorer,
    ) -> List[ToolCall]:
        left = []
        for index, turn in enumerate(turns):
            if turn.role != "assistant":
                continue
            calls = list(turn.tools_called or [])
            for expected in pinned.get(index, []):
                best = max(
                    calls,
                    key=lambda call: scorer._match_score(expected, call),
                    default=None,
                )
                if best is not None and scorer._match_score(expected, best) > 0:
                    calls.remove(best)
            left.extend(calls)
        return left

    def _pinned_tools(
        self, turns: List[Turn], expected_tools: List[ToolCall]
    ) -> Dict[int, List[ToolCall]]:
        pinned: Dict[int, List[ToolCall]] = {}
        for tool in expected_tools:
            if tool.turn is None:
                continue
            if not 0 <= tool.turn < len(turns):
                self.error = (
                    f"Expected tool '{tool.name}' is pinned to turn "
                    f"{tool.turn}, but the conversation only has "
                    f"{len(turns)} turns."
                )
                raise MissingTestCaseParamsError(self.error)
            if turns[tool.turn].role != "assistant":
                self.error = (
                    f"Expected tool '{tool.name}' is pinned to turn "
                    f"{tool.turn}, which is not an assistant turn."
                )
                raise MissingTestCaseParamsError(self.error)
            pinned.setdefault(tool.turn, []).append(tool)
        return pinned

    def _reason(self) -> str:
        if len(self.check_scores) == 1:
            return self.check_scores[0]["reason"]
        failing = [c for c in self.check_scores if c["score"] < 1]
        passed = len(self.check_scores) - len(failing)
        lines = [
            f"The score is {self.score:.2f}: {passed} of "
            f"{len(self.check_scores)} checks passed."
        ]
        lines += [
            f"- {c['check']} ({c['score']:.2f}): {c['reason']}" for c in failing
        ]
        return "\n".join(lines)

    @property
    def __name__(self):
        return "Conversation Tool Correctness"
