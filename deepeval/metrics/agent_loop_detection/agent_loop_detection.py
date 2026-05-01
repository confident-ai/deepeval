import json
from typing import Optional, List, Union, Tuple, Dict

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import (
    initialize_model,
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.indicator import metric_progress_indicator


class AgentLoopDetectionMetric(BaseMetric):
    """Detects infinite loops and cyclical patterns in agent execution traces.

    Analyzes three sub-signals: tool call repetition, reasoning stagnation,
    and call graph cycles. Returns a score from 0.0 (severe looping) to
    1.0 (clean execution).
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        repetition_threshold: int = 3,
        similarity_threshold: float = 0.85,
        check_tool_repetition: bool = True,
        check_reasoning_stagnation: bool = True,
        check_call_graph_cycles: bool = True,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.repetition_threshold = repetition_threshold
        self.similarity_threshold = similarity_threshold
        self.check_tool_repetition = check_tool_repetition
        self.check_reasoning_stagnation = check_reasoning_stagnation
        self.check_call_graph_cycles = check_call_graph_cycles
        if model is not None:
            self.model, self.using_native_model = initialize_model(model)
            self.evaluation_model = self.model.get_model_name()
        else:
            self.model = None
            self.using_native_model = True
            self.evaluation_model = None
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
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
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                        _log_metric_to_confident=_log_metric_to_confident,
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
        tool_spans = [s for s in all_spans if s.get("type") == "tool"]
        llm_spans = [s for s in all_spans if s.get("type") == "llm"]

        rep_score, rep_reason = 1.0, "Tool repetition check skipped."
        if self.check_tool_repetition:
            rep_score, rep_reason = self._score_tool_repetition(tool_spans)

        stag_score, stag_reason = 1.0, "Reasoning stagnation check skipped."
        if self.check_reasoning_stagnation:
            stag_score, stag_reason = self._score_reasoning_stagnation(
                llm_spans
            )

        cycle_score, cycle_reason = 1.0, "Call graph cycles check skipped."
        if self.check_call_graph_cycles:
            cycle_score, cycle_reason = self._score_call_graph_cycles(all_spans)

        self.score_breakdown = {
            "tool_repetition": rep_score,
            "reasoning_stagnation": stag_score,
            "call_graph_cycles": cycle_score,
        }
        self.score = self._combine_scores(rep_score, stag_score, cycle_score)
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.success = self.score >= self.threshold

        reasons = []
        if self.check_tool_repetition and rep_score < 1.0:
            reasons.append(rep_reason)
        if self.check_reasoning_stagnation and stag_score < 1.0:
            reasons.append(stag_reason)
        if self.check_call_graph_cycles and cycle_score < 1.0:
            reasons.append(cycle_reason)

        if self.score == 1.0:
            self.reason = "No loop patterns detected."
        else:
            self.reason = " ".join([r for r in reasons if r])
            if not self.reason.strip():
                self.reason = "Loops detected but no explicit reason provided."

        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Tool Repetition Score: {rep_score} ({rep_reason})",
                f"Reasoning Stagnation Score: {stag_score} ({stag_reason})",
                f"Call Graph Cycles Score: {cycle_score} ({cycle_reason})",
                f"Combined Loop Score: {self.score}",
                f"Final Reason: {self.reason}",
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

    def _score_tool_repetition(self, tool_spans: list) -> Tuple[float, str]:
        if not tool_spans:
            return 1.0, "No tool spans found."

        tool_counts = {}
        for span in tool_spans:
            name = span.get("name", "")

            input_val = span.get("input", {})
            if isinstance(input_val, str):
                try:
                    input_val = json.loads(input_val)
                except Exception:
                    pass

            if isinstance(input_val, dict):
                args_tuple = tuple(
                    sorted((str(k), str(v)) for k, v in input_val.items())
                )
            else:
                args_tuple = (str(input_val),)

            call_hash = (name, args_tuple)
            tool_counts[call_hash] = tool_counts.get(call_hash, 0) + 1

        max_reps = max(tool_counts.values()) if tool_counts else 0
        if max_reps == 0:
            return 1.0, "No tool repetition."

        most_repeated_call = max(tool_counts.items(), key=lambda x: x[1])
        tool_name = most_repeated_call[0][0]
        count = most_repeated_call[1]

        if count >= self.repetition_threshold * 2:
            return (
                0.0,
                f"Tool '{tool_name}' called {count} times with identical arguments.",
            )
        elif count >= self.repetition_threshold:
            return (
                0.5,
                f"Tool '{tool_name}' called {count} times with identical arguments.",
            )

        return 1.0, "Tool calls are within acceptable repetition limits."

    def _score_call_graph_cycles(self, all_spans: list) -> Tuple[float, str]:
        if not all_spans or len(all_spans) < 2:
            return 1.0, "No call graph cycles detected."

        nodes = []
        for span in all_spans:
            typ = span.get("type", "unknown")
            name = span.get("name", "unnamed")
            nodes.append((typ, name))

        adj = {}
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if u == v:
                continue
            if u not in adj:
                adj[u] = set()
            adj[u].add(v)

        visited = set()
        rec_stack = set()
        cycle_path = []

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    path.append(neighbor)
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        has_cycle = False
        for node in adj.keys():
            if node not in visited:
                if dfs(node, cycle_path):
                    has_cycle = True
                    break

        if has_cycle:
            cycle_start_idx = cycle_path.index(cycle_path[-1])
            cycle_nodes = cycle_path[cycle_start_idx:]
            cycle_str = " -> ".join(
                [f"{typ}:{name}" for typ, name in cycle_nodes]
            )
            return 0.0, f"Cycle detected in execution path: {cycle_str}."

        return 1.0, "No execution cycles detected."

    def _score_reasoning_stagnation(self, llm_spans: list) -> Tuple[float, str]:
        if len(llm_spans) < 2:
            return (
                1.0,
                "Not enough LLM spans to check for reasoning stagnation.",
            )

        def get_bigrams(text: str) -> set:
            words = str(text).lower().split()
            return set(zip(words, words[1:]))

        max_overlap = 0.0
        stagnating_pair = (-1, -1)

        for i in range(len(llm_spans) - 1):
            out1 = llm_spans[i].get("output", "")
            out2 = llm_spans[i + 1].get("output", "")

            if not isinstance(out1, str) or not isinstance(out2, str):
                continue

            bg1 = get_bigrams(out1)
            bg2 = get_bigrams(out2)

            if not bg1 and not bg2:
                continue

            intersection = bg1.intersection(bg2)
            union = bg1.union(bg2)

            if union:
                jaccard = len(intersection) / len(union)
                if jaccard > max_overlap:
                    max_overlap = jaccard
                    stagnating_pair = (i, i + 1)

        if max_overlap >= self.similarity_threshold:
            if max_overlap > 0.95:
                return (
                    0.0,
                    f"Identical reasoning outputs at steps {stagnating_pair[0]} and {stagnating_pair[1]}.",
                )
            else:
                return (
                    0.5,
                    f"High reasoning overlap ({max_overlap:.2f}) at steps {stagnating_pair[0]} and {stagnating_pair[1]}.",
                )

        return 1.0, "No reasoning stagnation."

    def _combine_scores(
        self, rep_score: float, stag_score: float, cycle_score: float
    ) -> float:
        weights = 0.0
        total = 0.0

        if self.check_tool_repetition:
            weights += 0.40
            total += rep_score * 0.40
        if self.check_reasoning_stagnation:
            weights += 0.35
            total += stag_score * 0.35
        if self.check_call_graph_cycles:
            weights += 0.25
            total += cycle_score * 0.25

        if weights == 0.0:
            return 1.0

        return total / weights

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
        return "Agent Loop Detection"
