import contextvars
from typing import override
from pydantic_ai.agent import Agent

metric_collection_var = None


class PydanticAIAgent(Agent):
    def __init__(self, *args, metric_collection: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_collection = metric_collection

    @override
    def _run_span_end_attributes(self, *args, **kwargs):
        result = super()._run_span_end_attributes(*args, **kwargs)
        if self.metric_collection:
            result["metric_collection"] = self.metric_collection
        return result
