from typing import Optional
from .handler import CrewAIEventsListener
import deepeval

def instrumentator(api_key: Optional[str] = None):
    if api_key:
        deepeval.login_with_confident_api_key(api_key)
    
    CrewAIEventsListener()