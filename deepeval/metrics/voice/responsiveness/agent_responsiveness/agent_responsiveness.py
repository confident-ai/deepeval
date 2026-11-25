from typing import List, Optional

from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseExchangeRollUpMetric,
)
from deepeval.metrics.voice.responsiveness.failed_response.failed_response import (
    FailedResponseMetric,
)
from deepeval.metrics.voice.responsiveness.reprompt.reprompt import (
    RepromptMetric,
)
from deepeval.metrics.voice.responsiveness.silent_reply.silent_reply import (
    SilentReplyMetric,
)
from deepeval.metrics.voice.responsiveness.unexpected_call_end.unexpected_call_end import (
    UnexpectedCallEndMetric,
)


class AgentResponsivenessMetric(BaseExchangeRollUpMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection

    def _parts(self) -> List[BaseDefectMetric]:
        return [
            FailedResponseMetric(
                use_speech_detection=self.use_speech_detection
            ),
            SilentReplyMetric(),
            UnexpectedCallEndMetric(),
            RepromptMetric(use_speech_detection=self.use_speech_detection),
        ]

    @property
    def __name__(self):
        return "Agent Responsiveness"
