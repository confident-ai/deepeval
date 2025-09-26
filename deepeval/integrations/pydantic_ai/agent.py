import warnings

try:
    from pydantic_ai.agent import Agent

    is_pydantic_ai_installed = True
except:
    is_pydantic_ai_installed = False


class DeepEvalPydanticAIAgent(Agent):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "instrument_pydantic_ai is deprecated and will be removed in a future version. "
            "Please use the new ConfidentInstrumentationSettings instead. Docs: https://www.confident-ai.com/docs/integrations/third-party/pydantic-ai",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)
