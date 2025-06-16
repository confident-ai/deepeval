from deepeval.openai_agents.callback_handler import OpenAIAgentsCallbackHandler

def trace_openai_agents():
    from agents import add_trace_processor     
    add_trace_processor(OpenAIAgentsCallbackHandler())