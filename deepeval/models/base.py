from abc import ABC, abstractmethod
from typing import Any, Optional


class DeepEvalBaseModel(ABC):
    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        self.model_name = model_name
        self.model = self.load_model(*args, **kwargs)

    @abstractmethod
    def load_model(self, *args, **kwargs):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    @abstractmethod
    def _call(self, *args, **kwargs):
        """Runs the model to score / ourput the model predictions.

        Returns:
            A score or a list of results.
        """
        pass
