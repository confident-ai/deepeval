from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union
from deepeval.models.utils import parse_model_name
from dataclasses import dataclass


@dataclass
class DeepEvalModelData:
    supports_log_probs: Optional[bool] = None
    supports_multimodal: Optional[bool] = None
    supports_structured_outputs: Optional[bool] = None
    supports_json: Optional[bool] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    supports_temperature: Optional[bool] = True


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
    def __init__(self, model: Optional[str] = None, *args, **kwargs):
        self.name = parse_model_name(model)
        self.model = self.load_model()

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

    @abstractmethod
    def get_model_name(self, *args, **kwargs) -> str:
        return self.name

    def batch_generate(self, *args, **kwargs) -> List[str]:
        """Runs the model to output LLM responses.

        Returns:
            A list of strings.
        """
        raise NotImplementedError(
            "batch_generate is not implemented for this model"
        )

    # Capabilities
    def supports_log_probs(self) -> Union[bool, None]:
        return None

    def supports_temperature(self) -> Union[bool, None]:
        return None

    def supports_multimodal(self) -> Union[bool, None]:
        return None

    def supports_structured_outputs(self) -> Union[bool, None]:
        return None

    def supports_json_mode(self) -> Union[bool, None]:
        return None

    def generate_with_schema(self, *args, schema=None, **kwargs):
        if schema is not None:
            try:
                return self.generate(*args, schema=schema, **kwargs)
            except TypeError:
                pass  # this means provider doesn't accept schema kwarg
        return self.generate(*args, **kwargs)

    async def a_generate_with_schema(self, *args, schema=None, **kwargs):
        if schema is not None:
            try:
                return await self.a_generate(*args, schema=schema, **kwargs)
            except TypeError:
                pass
        return await self.a_generate(*args, **kwargs)


class DeepEvalBaseEmbeddingModel(ABC):
    def __init__(self, model: Optional[str] = None, *args, **kwargs):
        self.name = parse_model_name(model)
        self.model = self.load_model()

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
        return self.name
