from typing import Optional
from easy_eval.config import EvaluatorConfig


class GeneralConfigMeta(EvaluatorConfig.__class__):
    def __new__(cls, name, bases, classdict):
        return super().__new__(cls, name, bases, classdict)


class APIEndpointConfigMeta(GeneralConfigMeta):
    def __new__(cls, name, bases, classdict):
        return super().__new__(cls, name, bases, classdict)


class GeneralConfig(EvaluatorConfig, metaclass=GeneralConfigMeta):
    """A config common for all the other configs"""

    top_p: Optional[float] = 0.95
    n_samples: Optional[int] = 1
    temperature: Optional[float] = 0.1
    context_length: Optional[int] = 512
    max_generation_length: Optional[int] = 512

    # limit helps to limit the number of samples to evaluate on benchmarks
    # right now limit, limit_start are unused
    limit: Optional[int] = None
    limit_start: Optional[int] = 0

    # load the generation path from an existing json
    load_generations_path: Optional[str] = None
    save_generations: Optional[bool] = False
    save_generations_path: Optional[str] = "default_generations.json"
    metric_output_path: Optional[str] = "default_evaluations.json"

    def update(self, **kwargs) -> "GeneralConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"KeyError: {key}")
        return self


class APIEndpointConfig(GeneralConfig, metaclass=APIEndpointConfigMeta):
    """Config for APIs. Right now this follows only OpenAI and Endpoints with OpenAI Spec"""

    # TODO: Anthropic is not supported in APIEndpointConfig.

    openai_api_key: Optional[str] = "sk-deepeval-default-none-key"
    openai_api_base: Optional[str] = ""
    openai_api_organization: Optional[str] = ""
    model: Optional[str] = "text-davinci-003"

    def update(self, **kwargs) -> "GeneralConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"KeyError: {key}")
        return self
