from typing import List

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
    def _parts(self) -> List[BaseDefectMetric]:
        return [
            FailedResponseMetric(),
            SilentReplyMetric(),
            UnexpectedCallEndMetric(),
            RepromptMetric(),
        ]

    @property
    def __name__(self):
        return "Agent Responsiveness"
