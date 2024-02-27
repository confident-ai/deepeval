from abc import abstractmethod
from typing import Optional, List, Union

from deepeval.test_case import LLMTestCase
from deepeval.models.base import DeepEvalBaseLLM


class BaseSynthesizer:
    synthesizer_model: Optional[str] = None

    @property
    def model(self) -> float:
        return self._model

    @model.setter
    def model(self, model: Optional[Union[str, DeepEvalBaseLLM]] = None):
        self._model = model

    @abstractmethod
    def synthesize(self, text: str, *args, **kwargs) -> List[LLMTestCase]:
        raise NotImplementedError
