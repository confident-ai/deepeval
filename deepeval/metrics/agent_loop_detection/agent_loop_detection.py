import json
from difflib import SequenceMatcher
from typing import Optional, List, Tuple, Dict

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.indicator import metric_progress_indicator

# Common stop words and agent boilerplate phrases that inflate Jaccard similarity
# without signalling true reasoning stagnation.
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "i",
    "will",
    "now",
    "based",
    "on",
    "information",
    "provided",
    "to",
    "of",
    "in",
    "and",
    "that",
    "this",
    "with",
    "for",
    "it",
    "my",
    "next",
    "step",
    "going",
    "so",
    "do",
    "be",
    "have",
    "has",
    "not",
    "but",
    "as",
    "or",
    "from",
    "at",
    "by",
    "about",
    "above",
    "below",
    "up",
    "its",
    "let",
}


class AgentLoopDetectionMetric(BaseMetric):
    """Detects infinite loops and cyclical patterns in agent execution traces.

    Analyzes three independent sub-signals and returns a weighted score from
    0.0 (severe looping) to 1.0 (clean execution):

    1. **Tool Call Repetition** — Counts identical ``(name, args)`` tool
       invocations.  Score degrades at ``repetition_threshold`` (0.5) and
       at ``2 × repetition_threshold`` (0.0).

    2. **Reasoning Stagnation** — Compares consecutive LLM-span outputs
       using *both* bigram Jaccard similarity *and* ``SequenceMatcher``
       ratio (stdlib ``difflib``).  Taking the maximum of the two catches
       stagnation whether the wording is literally repeated or merely
       reordered ("I will now search" vs "Let me search now").  Common
       stop words are stripped before comparison to prevent boilerplate
       from inflating scores.

    3. **Call Graph Cycles** — DFS on the nested ``children`` tree.  A
       cycle is flagged when a span with the same ``type:name:input_hash``
       label appears twice on the same root-to-leaf ancestry path.
       Including a truncated input hash in the label reduces false
       positives when two structurally different spans happen to share a
       name (see *Limitations* in the metric documentation).

    Design decisions
    ~~~~~~~~~~~~~~~~
    * **Fully deterministic** — no LLM / API key required.  This is
      intentional: the metric is designed to run in production pipelines
      at zero cost and zero latency.
    * **No ``model`` parameter** — because every sub-signal is computed
      with deterministic algorithms, accepting a ``model`` argument would
      be misleading.  A future LLM-as-judge stagnation mode could be added
      behind a feature flag if semantic comparison proves necessary.

    Limitations
    ~~~~~~~~~~~
    * **Cycle detection** relies on ``type:name:input_hash`` identity.
      Two genuinely different spans that share the same type, name, *and*
      a truncated input hash could still be false-positived.  The trace
      dict (``_trace_dict``) does not expose span UUIDs, so this is the
      best available heuristic.
    * **Stagnation detection** uses structural text similarity (bigram
      Jaccard + ``SequenceMatcher``).  It will miss semantically identical
      outputs that are worded very differently.  An LLM-as-judge mode
      would solve this but would sacrifice the deterministic / zero-cost
      properties.
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
            cycle_score, cycle_reason = self._score_call_graph_cycles(
                test_case._trace_dict
            )

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

    # ------------------------------------------------------------------
    # Call graph cycle detection
    # ------------------------------------------------------------------

    @staticmethod
    def _input_hash(span: Dict) -> str:
        """Return a short, stable hash of a span's input for label
        disambiguation.

        Including the input in the DFS label drastically reduces false
        positives when two structurally different spans share a
        ``type:name`` pair (e.g. two different agents both named
        ``"planner"``).  We truncate to 64 chars to keep labels readable
        in cycle-path messages.
        """
        raw = span.get("input", "")
        if isinstance(raw, dict):
            try:
                raw = json.dumps(raw, sort_keys=True)
            except (TypeError, ValueError):
                raw = str(raw)
        return str(raw)[:64]

    def _score_call_graph_cycles(
        self, trace_dict: Optional[Dict]
    ) -> Tuple[float, str]:
        """Detect cycles in the real parent→child call graph.

        Traverses the nested ``children`` tree using DFS.  A cycle is
        flagged when a span's ``type:name:input_hash`` label appears
        twice on the same root-to-leaf ancestry path — meaning the agent
        genuinely called itself (or a transitive ancestor) recursively.

        **Why input_hash is included:** Without it, two genuinely
        different spans that happen to share the same ``type:name`` (e.g.
        ``agent:planner`` at the root and a delegated ``agent:planner``
        with different input deeper in the tree) would be false-positived.
        Incorporating a truncated input hash makes the label specific
        enough to avoid this while still detecting true recursive loops
        (which, by definition, pass the same or similar input back).

        **Limitation:** If the trace dict exposed span UUIDs we could
        use exact identity.  ``create_nested_spans_dict`` strips them,
        so ``type:name:input_hash`` is the best available heuristic.

        Sequential repetition (the same tool appearing multiple times at
        sibling positions) is intentionally NOT flagged here; that is the
        job of ``_score_tool_repetition``.
        """
        if not trace_dict:
            return 1.0, "No call graph cycles detected."

        cycle_path: List[str] = []

        def _label(span: Dict) -> str:
            return (
                f"{span.get('type', 'unknown')}"
                f":{span.get('name', 'unnamed')}"
                f":{self._input_hash(span)}"
            )

        def dfs(span: Dict, ancestor_labels: List[str]) -> bool:
            """Return True as soon as a cycle is found, populating
            ``cycle_path`` with the offending ancestry chain."""
            label = _label(span)

            if label in ancestor_labels:
                # Found a back-edge: report the cycle path
                cycle_start = ancestor_labels.index(label)
                cycle_path.extend(ancestor_labels[cycle_start:])
                cycle_path.append(label)
                return True

            ancestor_labels.append(label)
            for child in span.get("children", []):
                if dfs(child, ancestor_labels):
                    return True
            ancestor_labels.pop()
            return False

        has_cycle = dfs(trace_dict, [])

        if has_cycle:
            # Strip the input_hash from the display for readability
            display_path = []
            for label in cycle_path:
                parts = label.split(":", 2)
                display_path.append(f"{parts[0]}:{parts[1]}")
            cycle_str = " -> ".join(display_path)
            return 0.0, f"Cycle detected in execution path: {cycle_str}."

        return 1.0, "No execution cycles detected."

    # ------------------------------------------------------------------
    # Reasoning stagnation detection
    # ------------------------------------------------------------------

    def _score_reasoning_stagnation(self, llm_spans: list) -> Tuple[float, str]:
        """Compare consecutive LLM outputs using two complementary
        similarity signals and take the **maximum**:

        1. **Bigram Jaccard** — fast bag-of-bigrams overlap after
           stop-word removal.  Good at catching literal repetition.
        2. **SequenceMatcher ratio** (``difflib``) — sequence-aware
           comparison that catches reordered but semantically identical
           text ("I will now search" ≈ "Let me search now").

        Taking the max ensures we flag stagnation regardless of whether
        the agent repeats itself verbatim or merely shuffles its phrasing.

        Outputs shorter than 20 meaningful words (after stop-word
        removal) are skipped — Jaccard is meaningless at that scale.
        """
        if len(llm_spans) < 2:
            return (
                1.0,
                "Not enough LLM spans to check for reasoning stagnation.",
            )

        def _clean_words(text: str) -> List[str]:
            """Lowercase, strip stop words, drop short tokens."""
            return [
                w
                for w in str(text).lower().split()
                if w not in _STOP_WORDS and len(w) > 2
            ]

        def _bigram_jaccard(words_a: List[str], words_b: List[str]) -> float:
            if len(words_a) < 2 or len(words_b) < 2:
                return 0.0
            bg_a = set(zip(words_a, words_a[1:]))
            bg_b = set(zip(words_b, words_b[1:]))
            union = bg_a | bg_b
            if not union:
                return 0.0
            return len(bg_a & bg_b) / len(union)

        def _sequence_ratio(text_a: str, text_b: str) -> float:
            """SequenceMatcher ratio — order-sensitive but resilient to
            small insertions/deletions."""
            return SequenceMatcher(None, text_a, text_b).ratio()

        max_overlap = 0.0
        stagnating_pair = (-1, -1)

        for i in range(len(llm_spans) - 1):
            out1 = llm_spans[i].get("output", "")
            out2 = llm_spans[i + 1].get("output", "")

            if not isinstance(out1, str) or not isinstance(out2, str):
                continue

            words1 = _clean_words(out1)
            words2 = _clean_words(out2)

            # Skip pairs where either output is too short for meaningful
            # comparison — Jaccard on < 20 words is noisy.
            if len(words1) < 20 or len(words2) < 20:
                continue

            jaccard = _bigram_jaccard(words1, words2)

            # SequenceMatcher runs on the cleaned word list (joined) so
            # it's also stop-word-free.
            seq_ratio = _sequence_ratio(" ".join(words1), " ".join(words2))

            # Take the maximum of both signals — catches both literal
            # repetition (Jaccard) and reordered repetition (SequenceMatcher).
            similarity = max(jaccard, seq_ratio)

            if similarity > max_overlap:
                max_overlap = similarity
                stagnating_pair = (i, i + 1)

        if max_overlap >= self.similarity_threshold:
            if max_overlap > 0.95:
                return (
                    0.0,
                    f"Identical reasoning outputs at steps "
                    f"{stagnating_pair[0]} and {stagnating_pair[1]}.",
                )
            else:
                return (
                    0.5,
                    f"High reasoning overlap ({max_overlap:.2f}) at steps "
                    f"{stagnating_pair[0]} and {stagnating_pair[1]}.",
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
