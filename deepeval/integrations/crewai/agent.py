try:
    from crewai.agent import Agent as CrewAIAgent
    crewai_installed = True
except:
    crewai_installed = False

from typing import Optional, List
from deepeval.metrics import BaseMetric

def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI Agent is not installed. Please install it with `pip install crewai`."
        )   

class PatchedAgent(CrewAIAgent):
    def __init__(
        self,
        *args,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        is_crewai_installed()
        self.metric_collection = metric_collection
        self.metrics = metrics
        super().__init__(*args, **kwargs)