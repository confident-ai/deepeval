from typing import Optional
import deepeval

try:
    from crewai.crew import Crew
    from crewai.llm import LLM
    from crewai.agent import Agent
    
    crewai_installed = True
except:
    crewai_installed = False

def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )
    
from deepeval.tracing import observe

def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)

    Crew.kickoff = observe(Crew.kickoff)
    LLM.call = observe(LLM.call)
    Agent.execute_task = observe(Agent.execute_task)