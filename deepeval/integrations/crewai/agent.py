from crewai.agent import Agent as CrewAIAgent
from typing import Optional, List
from deepeval.metrics import BaseMetric

class Agent(CrewAIAgent):
    def __init__(
        self,
        *args,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metric_collection = metric_collection
        self.metrics = metrics