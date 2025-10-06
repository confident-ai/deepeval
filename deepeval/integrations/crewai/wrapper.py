from crewai.llm import LLM
from crewai.crew import Crew
from functools import wraps
from deepeval.tracing.tracing import Observer

def wrap_crew_kickoff():
    original_kickoff = Crew.kickoff
    
    @wraps(original_kickoff)
    def wrapper(self, *args, **kwargs):
        with Observer(span_type="crew", func_name="kickoff"):
            result = original_kickoff(self, *args, **kwargs)
        
        return result
    Crew.kickoff = wrapper 


def wrap_llm_call():
    original_llm_call = LLM.call
    
    @wraps(original_llm_call)
    def wrapper(self, *args, **kwargs):
        with Observer(span_type="llm", func_name="call", observe_kwargs={"model": "temp_model"}):
            result = original_llm_call(self, *args, **kwargs)
        return result
    LLM.call = wrapper