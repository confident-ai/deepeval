from crewai.agent import Agent as CrewAIAgent
from typing import Optional, Dict, Any
import weakref

class AgentRegistry:
    """Global registry to track CrewAI agents and their metric collections."""
    
    def __init__(self):
        self._agent_metric_mapping: Dict[int, str] = {}
        self._agent_instances: Dict[int, weakref.ref] = {}
    
    def register_agent(self, agent: CrewAIAgent, metric_collection: Optional[str] = None):
        """Register a CrewAI agent with its metric collection."""
        agent_id = id(agent)
        self._agent_metric_mapping[agent_id] = metric_collection
        # Use weakref to avoid keeping agents alive in memory
        self._agent_instances[agent_id] = weakref.ref(agent, self._cleanup_agent)
    
    def get_metric_collection(self, agent: CrewAIAgent) -> Optional[str]:
        """Get the metric collection for a given agent."""
        agent_id = id(agent)
        return self._agent_metric_mapping.get(agent_id)
    
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
            del self._agent_instances[agent_id]
    
    def get_all_agents(self) -> Dict[int, Optional[str]]:
        """Get all registered agents and their metric collections."""
        return self._agent_metric_mapping.copy()

# Global registry instance
agent_registry = AgentRegistry()

class Agent(CrewAIAgent):
    def __init__(self, *args, metrics_collection: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Register this agent instance with its metric collection
        agent_registry.register_agent(self, metrics_collection)
    
    @property
    def metric_collection(self) -> Optional[str]:
        """Get the metric collection for this agent."""
        return agent_registry.get_metric_collection(self)