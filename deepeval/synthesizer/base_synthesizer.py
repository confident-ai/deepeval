from abc import abstractmethod
from typing import Optional, Dict, List

from deepeval.test_case import LLMTestCase


class BaseSynthesizer:
    @abstractmethod
    def synthesize(self, text:str, *args, **kwargs) -> List[LLMTestCase]:
        raise NotImplementedError

