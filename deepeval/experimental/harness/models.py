from typing import Optional, Union, List 

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.experimental.harness.config import (
    GeneralConfig,
    APIEndpointConfig,
)

try:
    from easy_eval.harness import HarnessTask, HarnessModels
except ImportError as error:
    raise ImportError(
        f"Error: {error}"
        "easy_eval is not found."
        "You can install it using: pip install easy-evaluator"
    )

class DeepEvalHarnessModel(DeepEvalBaseLLM):
    def __init__(
        self, model_name_or_path: str, model_backend: str, config: Optional[Union[GeneralConfig, APIEndpointConfig]] = None, **kwargs
    ) -> None:
        self.model_name_or_path, self.model_backend = (
            model_name_or_path,
            model_backend,
        )
        if config is None:
            self.config = APIEndpointConfig() if model_backend in ["openai"] else GeneralConfig()
        else:
            self.config = config
            
        self.additional_params = kwargs
        super().__init__(
            model_name=model_name_or_path, **self.additional_params
        )

    def load_model(self, *args, **kwargs):
        self.model = HarnessModels(
            model_name_or_path=self.model_name_or_path,
            model_backend=self.model_backend,
            config=self.config,
            **self.additional_params,
        )
        return self.model.lm

    def generate(self, tasks: Union[List[str], HarnessTask], *args, **kwargs) -> str:
        responses, task_dict, task_metadata = self.model.generate(
            tasks=tasks, return_task_metadata=True
        )
        self.task_dict, self.task_metadata = task_dict, task_metadata
        return responses