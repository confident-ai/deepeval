from abc import ABC, abstractmethod
from typing import Any, Optional, List
from deepeval.models.utils import parse_model_name


class DeepEvalBaseModel(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = model_name
        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs) -> "DeepEvalBaseModel":
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    @abstractmethod
    def _call(self, *args, **kwargs):
        """Runs the model to score / output the model predictions.

        Returns:
            A score or a list of results.
        """
        pass


class DeepEvalBaseLLM(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = parse_model_name(model_name)
        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs) -> "DeepEvalBaseLLM":
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

    @abstractmethod
    async def a_generate(self, *args, **kwargs) -> str:
        """Runs the model to output LLM response.

        Returns:
            A string.
        """
        pass

    def batch_generate(self, *args, **kwargs) -> List[str]:
        """Runs the model to output LLM responses.

        Returns:
            A list of strings.
        """
        raise NotImplementedError(
            "batch_generate is not implemented for this model"
        )

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        pass


class DeepEvalBaseMLLM(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = parse_model_name(model_name)

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Runs the model to output MLLM response.

        Returns:
            A string.
        """
        pass

    @abstractmethod
    async def a_generate(self, *args, **kwargs) -> str:
        """Runs the model to output MLLM response.

        Returns:
            A string.
        """
        pass

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        pass


class DeepEvalBaseEmbeddingModel(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = parse_model_name(model_name)

        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs) -> "DeepEvalBaseEmbeddingModel":
        """Loads a model, that will be responsible for generating text embeddings.

        Returns:
            A model object
        """
        pass

    @abstractmethod
    def embed_text(self, *args, **kwargs) -> List[float]:
        """Runs the model to generate text embeddings.

        Returns:
            A list of float.
        """
        pass

    @abstractmethod
    async def a_embed_text(self, *args, **kwargs) -> List[float]:
        """Runs the model to generate text embeddings.

        Returns:
            A list of list of float.
        """
        pass

    @abstractmethod
    def embed_texts(self, *args, **kwargs) -> List[List[float]]:
        """Runs the model to generate list of text embeddings.

        Returns:
            A list of float.
        """
        pass

    @abstractmethod
    async def a_embed_texts(self, *args, **kwargs) -> List[List[float]]:
        """Runs the model to generate list of text embeddings.

        Returns:
            A list of list of float.
        """
        pass

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        pass
