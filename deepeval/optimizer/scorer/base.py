from abc import ABC, abstractmethod
from typing import Union, List

from deepeval.optimizer.types import PromptConfiguration, ScoreVector
from deepeval.dataset.golden import Golden, ConversationalGolden

ModuleId = str


class BaseScorer(ABC):
    """
    Base scorer contract used by optimization runners.

    Runners call into this adapter to:
    - compute scores per-instance on some subset (score_on_pareto),
    - compute minibatch means for selection and acceptance,
    - generate feedback text used by the Rewriter.
    """

    # Sync
    @abstractmethod
    def score_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector:
        """Return per-instance scores on D_pareto."""
        raise NotImplementedError

    @abstractmethod
    def score_minibatch(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        """Return average score μ on a minibatch from D_feedback."""
        raise NotImplementedError

    @abstractmethod
    def get_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        """Return μ_f text for the module (metric.reason + traces, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId:
        """Pick a module to mutate."""
        raise NotImplementedError

    # Async
    @abstractmethod
    async def a_score_pareto(
        self,
        prompt_configuration: PromptConfiguration,
        d_pareto: Union[List[Golden], List[ConversationalGolden]],
    ) -> ScoreVector:
        raise NotImplementedError

    @abstractmethod
    async def a_score_minibatch(
        self,
        prompt_configuration: PromptConfiguration,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    async def a_get_minibatch_feedback(
        self,
        prompt_configuration: PromptConfiguration,
        module: ModuleId,
        minibatch: Union[List[Golden], List[ConversationalGolden]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def a_select_module(
        self, prompt_configuration: PromptConfiguration
    ) -> ModuleId:
        raise NotImplementedError
