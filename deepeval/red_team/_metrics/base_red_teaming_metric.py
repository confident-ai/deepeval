from typing import List

from deepeval.metrics import BaseMetric
from deepeval.test_case import SingleTurnParams


class BaseRedTeamingMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]
