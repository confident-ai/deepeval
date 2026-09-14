import os
from typing import Optional
from deepeval.singleton import Singleton
from deepeval.progress_context import progress_context

DEFAULT_HALLUCINATION_MODEL = "vectara/hallucination_evaluation_model"


class HallucinationModel(metaclass=Singleton):
    def __init__(
        self,
        model_name: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ):
        """Load a CrossEncoder used for hallucination scoring.

        Args:
            model_name: Hugging Face CrossEncoder id. Defaults to
                ``vectara/hallucination_evaluation_model``.
            trust_remote_code: Forwarded to ``CrossEncoder``. Defaults to True
                for the default Vectara model (which requires custom code) and
                False for any other model unless the caller passes True.
            **kwargs: Additional arguments forwarded to
                ``sentence_transformers.CrossEncoder``.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "The 'sentence_transformers' library is required to use the HallucinationMetric."
            )
        # We use a smple cross encoder model
        model_name = (
            DEFAULT_HALLUCINATION_MODEL if model_name is None else model_name
        )
        if trust_remote_code is None:
            trust_remote_code = model_name == DEFAULT_HALLUCINATION_MODEL
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # TODO: add this progress context in the correct place
        with progress_context(
            "Downloading HallucinationEvaluationModel (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = CrossEncoder(
                model_name, trust_remote_code=trust_remote_code, **kwargs
            )
