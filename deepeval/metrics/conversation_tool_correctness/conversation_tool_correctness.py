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

    # Matching for Conversation Tool Correctness. It differs from
    # ToolCorrectnessMetric in that an expected tool's unset input parameters
    # or output are not compared, repeated identical calls each count, and
    # the reason names the tools that are missing or differ.

    def _generate_reason(self):
        tools_called_names = [
            tool_called.name for tool_called in self.tools_called
        ]
        expected_tools_names = [
            expected_tool.name for expected_tool in self.expected_tools
        ]

        type_mismatches = self._get_type_mismatches()

        if self.should_exact_match:
            return (
                f"{'Exact match' if self._calculate_exact_match_score() else 'Not an exact match'}: expected {expected_tools_names}, called {tools_called_names}."
                + (
                    f" Tool type mismatches: {type_mismatches}."
                    if type_mismatches
                    else ""
                )
                + " See details above."
            )

        elif self.should_consider_ordering:
            lcs, weighted_length = self._compute_weighted_lcs()
            if (
                len(self.tools_called) == len(self.expected_tools)
                and len(self.expected_tools) == 0
            ):
                score = 1.0
            elif len(self.expected_tools) == 0:
                score = 0.0
            else:
                score = weighted_length / len(self.expected_tools)
            missing, wrong_inputs, wrong_outputs = self._tool_call_issues()
            in_lcs = [tool.name for tool in lcs]
            out_of_order = []
            for name in expected_tools_names:
                if name in in_lcs:
                    in_lcs.remove(name)
                elif name not in missing and name not in wrong_outputs:
                    out_of_order.append(name)
            if score == 1:
                return f"Correct ordering: all expected tools {expected_tools_names} were called in the correct order."
            else:
                issues = self._issue_list(
                    missing, wrong_inputs, wrong_outputs, type_mismatches
                )
                if out_of_order:
                    after_missing = 1 if missing else 0
                    issues.insert(
                        after_missing, f"out-of-order tools {out_of_order}"
                    )
                if not issues:
                    issues.append("tools did not fully match")
                return f"Incorrect tool usage: {'; '.join(issues)}; expected {expected_tools_names}, called {tools_called_names}. See more details above."
        else:
            missing, wrong_inputs, wrong_outputs = self._tool_call_issues()
            if self._calculate_non_exact_match_score() == 1:
                return f"All expected tools {expected_tools_names} were called (order not considered)."
            else:
                issues = self._issue_list(
                    missing, wrong_inputs, wrong_outputs, type_mismatches
                )
                if not issues:
                    issues.append("tools did not fully match")
                return f"Incomplete tool usage: {'; '.join(issues)}; expected {expected_tools_names}, called {tools_called_names}. See more details above."

    def _tool_call_issues(self) -> Tuple[List[str], List[str], List[str]]:
        """Names of expected tools that were never called, and of those
        called with input parameters or output that differ, when those are
        evaluated. Each call is matched to one expected tool at most."""

        def inputs_match(expected: ToolCall, called: ToolCall) -> bool:
            return self._input_parameters_score(expected, called) == 1

        output_matches = self._output_matches

        unmatched = list(self.tools_called)
        missing, wrong_inputs, wrong_outputs = [], [], []
        for expected_tool in self.expected_tools:
            same_name = [
                called_tool
                for called_tool in unmatched
                if called_tool.name == expected_tool.name
            ]
            if not same_name:
                missing.append(expected_tool.name)
                continue
            match = next(
                (
                    called_tool
                    for called_tool in same_name
                    if inputs_match(expected_tool, called_tool)
                    and output_matches(expected_tool, called_tool)
                ),
                same_name[0],
            )
            unmatched.remove(match)
            if not inputs_match(expected_tool, match):
                wrong_inputs.append(expected_tool.name)
            if not output_matches(expected_tool, match):
                wrong_outputs.append(expected_tool.name)
        return missing, wrong_inputs, wrong_outputs

    def _issue_list(
        self,
        missing: List[str],
        wrong_inputs: List[str],
        wrong_outputs: List[str],
        type_mismatches: List[str],
    ) -> List[str]:
        issues = []
        if missing:
            issues.append(f"missing tools {missing}")
        if wrong_inputs:
            issues.append(f"different input parameters for {wrong_inputs}")
        if wrong_outputs:
            issues.append(f"different output for {wrong_outputs}")
        if type_mismatches:
            issues.append(f"tool type mismatches {type_mismatches}")
        if not self.expected_tools and self.tools_called:
            issues.append(
                "no tools were expected, but "
                f"{[tool.name for tool in self.tools_called]} were called"
            )
        return issues

    def _input_parameters_score(
        self, expected_tool: ToolCall, called_tool: ToolCall
    ) -> float:
        """Input parameters are only compared when they are evaluated and the
        expected tool sets them; an expected tool without them matches any
        call's."""
        if (
            ToolCallParams.INPUT_PARAMETERS not in self.evaluation_params
            or expected_tool.input_parameters is None
        ):
            return 1.0
        return self._compare_dicts(
            expected_tool.input_parameters, called_tool.input_parameters or {}
        )

    def _output_matches(
        self, expected_tool: ToolCall, called_tool: ToolCall
    ) -> bool:
        """Output is only compared when it is evaluated and the expected tool
        sets it."""
        return (
            ToolCallParams.OUTPUT not in self.evaluation_params
            or expected_tool.output is None
            or expected_tool.output == called_tool.output
        )

    def _calculate_exact_match_score(self) -> float:
        if len(self.tools_called) != len(self.expected_tools):
            return 0.0
        if (
            len(self.tools_called) == len(self.expected_tools)
            and len(self.expected_tools) == 0
        ):
            return 1.0
        for i in range(len(self.tools_called)):
            if self.tools_called[i].name != self.expected_tools[i].name:
                return 0.0
            if self.tools_called[i].type != self.expected_tools[i].type:
                return 0.0
            if (
                self._input_parameters_score(
                    self.expected_tools[i], self.tools_called[i]
                )
                != 1
            ):
                return 0.0
            if not self._output_matches(
                self.expected_tools[i], self.tools_called[i]
            ):
                return 0.0
        return 1.0

    def _calculate_non_exact_match_score(self) -> float:
        total_score = 0.0
        # Track matched calls by position: identical calls are equal, so a
        # set of calls would treat a repeated call as already used.
        matched_indices = set()
        for expected_tool in self.expected_tools:
            best_score = 0.0
            best_index = None
            for index, called_tool in enumerate(self.tools_called):
                if index in matched_indices:
                    continue
                if (
                    expected_tool.name == called_tool.name
                    and expected_tool.type == called_tool.type
                ):
                    match_score = self._input_parameters_score(
                        expected_tool, called_tool
                    )
                    if not self._output_matches(expected_tool, called_tool):
                        match_score = 0.0
                    if match_score > best_score:
                        best_score = match_score
                        best_index = index
            if best_index is not None:
                total_score += best_score
                matched_indices.add(best_index)
        return (
            1.0
            if not self.expected_tools and not self.tools_called
            else (
                0.0
                if not self.expected_tools
                else total_score / len(self.expected_tools)
            )
        )

    def _compute_weighted_lcs(self) -> Tuple[List[ToolCall], float]:
        m, n = len(self.expected_tools), len(self.tools_called)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                expected_tool, called_tool = (
                    self.expected_tools[i - 1],
                    self.tools_called[j - 1],
                )
                if (
                    expected_tool.name != called_tool.name
                    or expected_tool.type != called_tool.type
                ):
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    continue
                score = self._input_parameters_score(expected_tool, called_tool)
                if not self._output_matches(expected_tool, called_tool):
                    score = 0.0
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1] + score if score > 0 else 0,
                )
        i, j, total_score = m, n, 0.0
        lcs = []
        while i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j]:
                i -= 1
            elif dp[i][j] == dp[i][j - 1]:
                j -= 1
            else:
                lcs.append(self.expected_tools[i - 1])
                total_score += dp[i][j] - dp[i - 1][j - 1]
                i, j = i - 1, j - 1
        return lcs[::-1], total_score


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
        """Every assistant call, minus one call per tool pinned to its turn,
        so an unpinned tool can be met on any turn but a pinned call is not
        counted twice."""
        left = []
        for index, turn in enumerate(turns):
            if turn.role != "assistant":
                continue
            calls = list(turn.tools_called or [])
            for expected in pinned.get(index, []):
                candidates = [
                    call
                    for call in calls
                    if call.name == expected.name and call.type == expected.type
                ]
                if not candidates:
                    continue
                match = next(
                    (
                        call
                        for call in candidates
                        if scorer._input_parameters_score(expected, call) == 1
                        and scorer._output_matches(expected, call)
                    ),
                    candidates[0],
                )
                calls.pop(next(i for i, c in enumerate(calls) if c is match))
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
