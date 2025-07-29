try:
    from pydantic_ai import Agent
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
        print("args--------------")
        print(args)
        print("kwargs--------------")
        print(kwargs)
        result = await original_run(*args, **kwargs)
        print("result--------------")
        print(result)
        return result
    
    # Apply the patch
    Agent.run = patched_run