from typing import List, Optional

from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class SimulationPathMetric(BaseVoiceMetric):
    """Checks where a simulation graph conversation ended.

    Passes when the conversation reached an ending that is not a failure
    without the simulated user saying any line marked `fails=True`.
    """

    def __init__(
        self,
        expected_endings: Optional[List[str]] = None,
        threshold: Optional[float] = 1.0,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.expected_endings = expected_endings

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        metadata = test_case.metadata or {}
        if "Simulation path" not in metadata:
            return (
                None,
                "The test case was not produced by a simulation graph.",
                {},
            )
        ending = metadata.get("Simulation ending")
        failed_nodes = metadata.get("Failed nodes") or []
        breakdown = {
            "ending": ending,
            "failed_nodes": failed_nodes,
            "path": metadata["Simulation path"],
        }
        if ending is None:
            return (
                0.0,
                "The conversation stopped before reaching an ending.",
                breakdown,
            )
        if ending == "off_script":
            return 0.0, "The conversation went off script.", breakdown
        if failed_nodes:
            return (
                0.0,
                "The agent's replies led the caller to: "
                + ", ".join(failed_nodes)
                + ".",
                breakdown,
            )
        if self.expected_endings and ending not in self.expected_endings:
            return (
                0.0,
                f"The conversation ended on '{ending}', which is not one of "
                f"the expected endings: {', '.join(self.expected_endings)}.",
                breakdown,
            )
        return (
            1.0,
            f"The conversation reached the ending '{ending}'.",
            breakdown,
        )

    @property
    def __name__(self):
        return "Simulation Path"
