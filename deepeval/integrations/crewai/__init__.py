from .patchers import CrewAILogger

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

def instrumentator():
    crewai_logger.patch_crewai_LLM("call")