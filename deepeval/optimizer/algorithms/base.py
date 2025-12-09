from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.optimizer.scorer.base import BaseScorer
from deepeval.prompt.prompt import Prompt
from deepeval.dataset.golden import Golden, ConversationalGolden


class BaseAlgorithm(ABC):
    name: str
    optimizer_model: DeepEvalBaseLLM
    scorer: BaseScorer

    @abstractmethod
    def execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, Dict]:
        raise NotImplementedError

    @abstractmethod
    async def a_execute(
        self,
        prompt: Prompt,
        goldens: Union[List[Golden], List[ConversationalGolden]],
    ) -> Tuple[Prompt, Dict]:
        raise NotImplementedError
