from typing import Optional, Dict, Any, List
import weakref
from deepeval.metrics import BaseMetric
from deepeval.telemetry import capture_tracing_integration

try:
    from crewai.agent import Agent as CrewAIAgent

    crewai_installed = True
except:
    crewai_installed = False


def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )


class AgentRegistry:
    """Global registry to track CrewAI agents, their metric collections, and metrics."""

    def __init__(self):
        is_crewai_installed()
        self._agent_metric_mapping: Dict[int, str] = {}
        self._agent_metrics_mapping: Dict[int, List[BaseMetric]] = {}
        self._agent_instances: Dict[int, weakref.ref] = {}

    def register_agent(
        self,
        agent: CrewAIAgent,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ):
        """Register a CrewAI agent with its metric collection and metrics."""
        agent_id = id(agent)
        self._agent_metric_mapping[agent_id] = metric_collection
        self._agent_metrics_mapping[agent_id] = metrics or []
        self._agent_instances[agent_id] = weakref.ref(
            agent, self._cleanup_agent
        )

    def get_metric_collection(self, agent: CrewAIAgent) -> Optional[str]:
        """Get the metric collection for a given agent."""
        agent_id = id(agent)
        return self._agent_metric_mapping.get(agent_id)

    def get_metrics(self, agent: CrewAIAgent) -> List[BaseMetric]:
        agent_id = id(agent)
        return self._agent_metrics_mapping.get(agent_id, [])

    def _cleanup_agent(self, weak_ref):
        """Clean up agent references when they're garbage collected."""
        # Find and remove the agent_id for this weak reference
        agent_id = None
        for aid, ref in self._agent_instances.items():
            if ref == weak_ref:
                agent_id = aid
                break

        if agent_id:
            del self._agent_metric_mapping[agent_id]
            del self._agent_metrics_mapping[agent_id]
            del self._agent_instances[agent_id]

    def get_all_agents(self) -> Dict[int, Optional[str]]:
        """Get all registered agents and their metric collections."""
        return self._agent_metric_mapping.copy()


# Global registry instance
agent_registry = AgentRegistry()


class Agent(CrewAIAgent):
    def __init__(
        self,
        *args,
        metric_collection: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs
    ):
        with capture_tracing_integration("crewai.agent.Agent"):
            is_crewai_installed()
            super().__init__(*args, **kwargs)
            # Register this agent instance with its metric collection
            agent_registry.register_agent(self, metric_collection, metrics)

    @property
    def metric_collection(self) -> Optional[str]:
        """Get the metric collection for this agent."""
        return agent_registry.get_metric_collection(self)

    @property
    def metrics(self) -> List[BaseMetric]:
        """Get the list of metrics for this agent."""
        return agent_registry.get_metrics(self)
