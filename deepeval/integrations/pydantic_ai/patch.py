import json
from deepeval.integrations.pydantic_ai import Agent as PatchedAgent
from opentelemetry.trace import NoOpTracer

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentedModel
    pydantic_ai_installed = True
except:
    pydantic_ai_installed = False

def is_pydantic_ai_installed():
    if not pydantic_ai_installed:
        raise ImportError("Pydantic AI is not installed. Please install it with `pip install pydantic-ai`.")


def safe_patch_agent_run_method():
    is_pydantic_ai_installed()
    original_run = Agent.run
    
    # define patched run method
    async def patched_run(*args, **kwargs):
        if isinstance(args[0], PatchedAgent):
            model_used = args[0]._get_model(kwargs.get('model', None))

            if isinstance(model_used, InstrumentedModel):
                instrumentation_settings = model_used.settings
                tracer = model_used.settings.tracer
            else:
                instrumentation_settings = None
                tracer = NoOpTracer()
        with tracer.start_as_current_span('confident_ai evaluation') as run_span:
            
            result = await original_run(*args, **kwargs)
            run_span.set_attribute('confident_ai.metric_collection', args[0].metric_collection)
            run_span.set_attribute('confident_ai.llm_test_case.input', str(args[1]))
            run_span.set_attribute('confident_ai.llm_test_case.actual_output', str(result.output))
        return result
    
    # Apply the patch
    Agent.run = patched_run