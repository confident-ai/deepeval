from .patchers import CrewAILogger
from typing import Optional
import deepeval

try:
    from crewai import LLM
    crewai_installed = True
except:
    crewai_installed = False

def is_crewai_installed():
    if not crewai_installed:
        raise ImportError(
            "CrewAI is not installed. Please install it with `pip install crewai`."
        )

crewai_logger = CrewAILogger()

def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)
        
    crewai_logger.patch_crewai_LLM("call")