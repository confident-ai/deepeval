import functools
from deepeval.tracing.tracing import Observer
try:
    from pydantic_ai.agent import Agent
    pydantic_ai_installed = True
except:
    pydantic_ai_installed = True

def _patch_agent_run():
    original_run = Agent.run

    @functools.wraps(original_run)
    async def wrapper(*args, **kwargs):
        with Observer(
            span_type="agent",
            func_name="Agent",
            function_kwargs={"input": args[1]},
        ) as observer:
            result = await original_run(*args, **kwargs)
            observer.result = result.output
        return result

    Agent.run = wrapper

def _patch_agent_run_sync():
    original_run = Agent.run_sync

    @functools.wraps(original_run)
    def wrapper(*args, **kwargs):
        with Observer(
            span_type="agent",
            func_name="Agent",
            function_kwargs={"input": args[1]},
        ) as observer:
            result = original_run(*args, **kwargs)
            observer.result = result.output
        return result
    Agent.run_sync = wrapper

def patch_all():
    _patch_agent_run()
    _patch_agent_run_sync()

