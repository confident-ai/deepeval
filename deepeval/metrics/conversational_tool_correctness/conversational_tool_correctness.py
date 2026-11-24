"""Tool Correctness for conversations.

The test case's ``expected_tools`` are checked with the Tool Correctness
rules. An expected tool with ``turn`` set is pinned to that turn: the turn's
own ``tools_called`` are scored against the tools pinned to it. Expected tools
without ``turn`` are scored against the tools called on every other assistant
turn, in order. The score is the mean over these groups."""

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
    """Scores a conversation's tool calls against ``expected_tools`` with the
    Tool Correctness rules. ``ToolCall.turn`` pins an expected tool to one
    assistant turn; the rest are checked across the conversation. No LLM is
    used."""

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

        expected_tools = test_case.expected_tools
        pinned = self._pinned_tools(test_case.turns, expected_tools)
        unpinned = [tool for tool in expected_tools if tool.turn is None]

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            scorer = _ToolCallScorer(
                self.evaluation_params,
                self.should_exact_match,
                self.should_consider_ordering,
            )
            # Each group is (label, tools called, tools expected).
            groups: List[Tuple[str, List[ToolCall], List[ToolCall]]] = [
                (
                    f"Turn {index}",
                    test_case.turns[index].tools_called or [],
                    tools,
                )
                for index, tools in sorted(pinned.items())
            ]
            if unpinned or not pinned:
                groups.append(
                    (
                        (
                            "Rest of the conversation"
                            if pinned
                            else "Conversation"
                        ),
                        self._tools_called(
                            test_case.turns, skip=set(pinned.keys())
                        ),
                        unpinned,
                    )
                )

            self.group_scores = []
            for label, called, expected in groups:
                score, reason = scorer.score(called, expected)
                self.group_scores.append(
                    {"group": label, "score": score, "reason": reason}
                )
            self.tools_called = self._tools_called(test_case.turns)
            self.expected_tools = expected_tools

            score = sum(g["score"] for g in self.group_scores) / len(
                self.group_scores
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
                        f"{g['group']}: score {g['score']:.2f}. {g['reason']}"
                        for g in self.group_scores
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
    def _tools_called(
        turns: List[Turn], skip: Optional[set] = None
    ) -> List[ToolCall]:
        return [
            tool
            for index, turn in enumerate(turns)
            if turn.role == "assistant" and index not in (skip or set())
            for tool in (turn.tools_called or [])
        ]

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
                    f"{tool.turn}, but the conversation has {len(turns)} "
                    f"turns (indexes 0 to {len(turns) - 1})."
                )
                raise MissingTestCaseParamsError(self.error)
            if turns[tool.turn].role != "assistant":
                self.error = (
                    f"Expected tool '{tool.name}' is pinned to turn "
                    f"{tool.turn}, which is a '{turns[tool.turn].role}' turn. "
                    "'turn' is the index in 'turns' of an assistant turn."
                )
                raise MissingTestCaseParamsError(self.error)
            pinned.setdefault(tool.turn, []).append(tool)
        return pinned

    def _reason(self) -> str:
        if len(self.group_scores) == 1:
            return self.group_scores[0]["reason"]
        failing = [g for g in self.group_scores if g["score"] < 1]
        summary = (
            f"The score is {self.score:.2f}: "
            f"{len(self.group_scores) - len(failing)} of "
            f"{len(self.group_scores)} checks passed."
        )
        if not failing:
            return summary
        details = "\n".join(
            f"- {g['group']} ({g['score']:.2f}): {g['reason']}" for g in failing
        )
        return f"{summary}\n{details}"

    @property
    def __name__(self):
        return "Conversational Tool Correctness"
