from typing import Optional, Union, List 

from deepeval.models.base_model import DeepEvalBaseModel
from deepeval.experimental.harness.config import GeneralConfig, APIEndpointConfig


class DeepEvalHarnessModel(DeepEvalBaseModel):
    def __init__(self, model_name_or_path: str, model_backend: str, **kwargs) -> None:
        self.model_name_or_path, self.model_backend = model_name_or_path, model_backend
        self.additional_params = kwargs
        super().__init__(model_name=model_name_or_path, **self.additional_params)

        try:
            from easy_eval import HarnessEvaluator
        except ImportError as error:
            raise ImportError(
                f"Error: {error}"
                "easy_eval is not found."
                "You can install it using: pip install easy-evaluator"
            )
        self.evaluator = HarnessEvaluator(
            model_name_or_path=self.model_name_or_path, model_backend=self.model_backend, **self.additional_params
        )
        
    def load_model(self, *args, **kwargs):
        return self.evaluator.llm 
    
    def _call(self, tasks: List[str], config: Optional[Union[GeneralConfig, APIEndpointConfig]] = None):
        # TODO: Anthropic is not supported in APIEndpointConfig. 
        if config is None:
            self.config = APIEndpointConfig() if self.model_name_or_path == "openai" else GeneralConfig()
        else:
            self.config = config
        return self.evaluator.evaluate(tasks=tasks, config=self.config)
