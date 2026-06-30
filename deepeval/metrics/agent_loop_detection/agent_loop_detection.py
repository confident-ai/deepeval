from typing import Optional, Union, List, Literal
import hashlib
import json
from collections import Counter, defaultdict

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.agent_loop_detection.schema import (
    LoopDetectionVerdict,
    LoopTrigger,
)


class AgentLoopDetectionMetric(BaseMetric):
    """Detects infinite loops and cyclical patterns in agent traces.
    
    This metric analyzes agent execution traces to detect:
    - Tool call repetition (same tool + args called repeatedly)
    - Reasoning stagnation (LLM producing similar outputs)
    - Call cycles (A→B→C→A patterns)
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        repetition_threshold: int = 3,
        min_identical_args_ratio: float = 0.9,
        reasoning_stagnation_detector: Literal["ngram", "embedding"] = "ngram",
        similarity_threshold: float = 0.85,
        stall_steps: int = 5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.repetition_threshold = repetition_threshold
        self.min_identical_args_ratio = min_identical_args_ratio
        self.reasoning_stagnation_detector = reasoning_stagnation_detector
        self.similarity_threshold = similarity_threshold
        self.stall_steps = stall_steps
        self.model = model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.requires_trace = True

    def measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        check_llm_test_case_params(
            test_case, self._required_params, None, None, self, self.model
        )

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator
        ):
            verdict = self._detect_loops(test_case)
            self.score = verdict.score
            self.reason = verdict.reason
            self.loop_triggers = verdict.loop_triggers
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Loop Detection Score: {self.score}",
                    f"Reason: {self.reason}",
                    f"Triggers: {len(self.loop_triggers)}",
                ],
            )

        return self.score

    def _detect_loops(self, test_case: LLMTestCase) -> LoopDetectionVerdict:
        """Detect all loop patterns in the trace."""
        trace = test_case._trace_dict
        if not trace or "steps" not in trace:
            return LoopDetectionVerdict(
                score=1.0,
                reason="No trace available for loop detection",
                loop_triggers=[],
            )

        steps = trace["steps"]
        triggers = []

        # Detect tool call repetition
        tool_triggers = self._detect_tool_repeats(steps)
        triggers.extend(tool_triggers)

        # Detect reasoning stagnation
        stagnation_triggers = self._detect_reasoning_stagnation(steps)
        triggers.extend(stagnation_triggers)

        # Detect call cycles
        cycle_triggers = self._detect_call_cycles(steps)
        triggers.extend(cycle_triggers)

        # Calculate score
        if not triggers:
            score = 1.0
            reason = "No loops detected in agent trace"
        else:
            # Score based on severity
            severity = self._calculate_severity(triggers, len(steps))
            score = max(0.0, 1.0 - severity)
            reason = self._generate_reason(triggers)

        return LoopDetectionVerdict(
            score=score, reason=reason, loop_triggers=triggers
        )

    def _detect_tool_repeats(self, steps: List[dict]) -> List[LoopTrigger]:
        """Detect repeated tool calls with identical or nearly identical arguments."""
        triggers = []
        tool_calls = []

        for i, step in enumerate(steps):
            if "tool_name" in step and "tool_args" in step:
                tool_calls.append((i, step["tool_name"], step["tool_args"]))

        # Build fingerprints
        fingerprints = defaultdict(list)
        for idx, tool_name, args in tool_calls:
            fp = self._fingerprint_tool_call(tool_name, args)
            fingerprints[fp].append(idx)

        # Check for repetitions
        for fp, indices in fingerprints.items():
            if len(indices) >= self.repetition_threshold:
                # Extract tool name from first occurrence
                tool_name = tool_calls[indices[0]][1]
                triggers.append(
                    LoopTrigger(
                        type="tool_repeat",
                        tool=tool_name,
                        steps=indices,
                        args_fingerprint=fp,
                        description=f"{tool_name} called {len(indices)}x with identical args in steps {indices}",
                    )
                )

        return triggers

    def _fingerprint_tool_call(self, tool_name: str, args: dict) -> str:
        """Create a fingerprint for tool call detection."""
        # Canonicalize args
        canonical = json.dumps(args, sort_keys=True)
        combined = f"{tool_name}:{canonical}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _detect_reasoning_stagnation(self, steps: List[dict]) -> List[LoopTrigger]:
        """Detect stagnation in LLM reasoning outputs."""
        triggers = []
        reasoning_steps = []

        for i, step in enumerate(steps):
            if "llm_output" in step or "reasoning" in step:
                text = step.get("llm_output") or step.get("reasoning", "")
                reasoning_steps.append((i, text))

        # Sliding window n-gram overlap
        if self.reasoning_stagnation_detector == "ngram":
            window = 3
            for i in range(len(reasoning_steps) - window + 1):
                indices = [reasoning_steps[i + j][0] for j in range(window)]
                texts = [reasoning_steps[i + j][1] for j in range(window)]
                
                # Check similarity across window
                if self._ngram_similarity(texts) > self.similarity_threshold:
                    triggers.append(
                        LoopTrigger(
                            type="reasoning_stagnation",
                            steps=indices,
                            description=f"Reasoning stagnation detected in steps {indices} (n-gram similarity > {self.similarity_threshold})",
                        )
                    )

        return triggers

    def _ngram_similarity(self, texts: List[str]) -> float:
        """Calculate average pairwise n-gram similarity."""
        if len(texts) < 2:
            return 0.0

        def get_ngrams(text: str, n: int = 3) -> set:
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                ng1 = get_ngrams(texts[i])
                ng2 = get_ngrams(texts[j])
                if not ng1 or not ng2:
                    continue
                overlap = len(ng1 & ng2)
                union = len(ng1 | ng2)
                sim = overlap / union if union > 0 else 0.0
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _detect_call_cycles(self, steps: List[dict]) -> List[LoopTrigger]:
        """Detect cycles in tool call sequences (A→B→C→A)."""
        triggers = []
        tool_sequence = []

        for i, step in enumerate(steps):
            if "tool_name" in step:
                tool_sequence.append((i, step["tool_name"]))

        # Simple cycle detection: look for repeating patterns
        if len(tool_sequence) < 4:
            return triggers

        # Check for cycles up to length 5
        for cycle_len in range(2, 6):
            for i in range(len(tool_sequence) - 2 * cycle_len + 1):
                pattern = [t[1] for t in tool_sequence[i:i+cycle_len]]
                next_pattern = [t[1] for t in tool_sequence[i+cycle_len:i+2*cycle_len]]
                
                if pattern == next_pattern:
                    indices = [t[0] for t in tool_sequence[i:i+2*cycle_len]]
                    triggers.append(
                        LoopTrigger(
                            type="call_cycle",
                            steps=indices,
                            description=f"Call cycle detected: {' → '.join(pattern)} repeated in steps {indices}",
                        )
                    )

        return triggers

    def _calculate_severity(self, triggers: List[LoopTrigger], total_steps: int) -> float:
        """Calculate overall loop severity."""
        if not triggers:
            return 0.0

        severities = []
        for trigger in triggers:
            affected_ratio = len(trigger.steps) / total_steps
            type_weight = {
                "tool_repeat": 1.0,
                "call_cycle": 0.8,
                "reasoning_stagnation": 0.6,
            }.get(trigger.type, 0.5)
            severities.append(affected_ratio * type_weight)

        return min(1.0, sum(severities))

    def _generate_reason(self, triggers: List[LoopTrigger]) -> str:
        """Generate human-readable reason."""
        if not triggers:
            return "No loops detected"

        parts = []
        for trigger in triggers[:3]:  # Top 3 triggers
            parts.append(trigger.description)

        return "; ".join(parts)

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
