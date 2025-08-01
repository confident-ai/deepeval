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
        
        if not isinstance(args[0], PatchedAgent):
            return await original_run(*args, **kwargs)
       
        # get tracer from model
        model_used = args[0]._get_model(kwargs.get('model', None))
        if isinstance(model_used, InstrumentedModel):
            tracer = model_used.settings.tracer
        else:
            tracer = NoOpTracer()
        with tracer.start_as_current_span('confident agent run') as run_span:

            result = await original_run(*args, **kwargs)
            
            # agent attributes
            run_span.set_attribute('confident.span.type', 'agent')
            run_span.set_attribute('confident.agent.name', str(args[0].name))
            run_span.set_attribute('confident.agent.attributes.input', str(args[1]))
            run_span.set_attribute('confident.agent.attributes.output', str(result.output))
            
            # llm test case attributes
            if args[0].metric_collection:
                run_span.set_attribute('confident.span.metric_collection', args[0].metric_collection)
            
            run_span.set_attribute('confident.span.llm_test_case.input', str(args[1]))
            run_span.set_attribute('confident.span.llm_test_case.actual_output', str(result.output))
    
        return result
    
    # Apply the patch
    Agent.run = patched_run