from typing import Optional
from deepeval.models.base_model import DeepEvalBaseModel


def softmax(x):
    import numpy as np

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class AnswerRelevancyModel(DeepEvalBaseModel):
    def __init__(self, model_name: Optional[str] = None):
        model_name = (
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
            if model_name is None
            else model_name
        )
        super().__init__(model_name=model_name)

    def load_model(self):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def _call(self, text: str):
        """Runs the model to score the predictions.

        Args:
            text (str): Text, which can be output from a LLM or a simple input text.

        Returns:
            Answer relevancy score.
        """
        if not hasattr(self, "model") or self.model is None:
            self.model = self.load_model()
        return self.model.encode(text)


class CrossEncoderAnswerRelevancyModel(DeepEvalBaseModel):
    def __init__(self, model_name: Optional[str] = None):
        model_name = (
            "cross-encoder/nli-deberta-v3-base"
            if model_name is None
            else model_name
        )
        super().__init__(model_name)

    def load_model(self):
        """Loads a model, that will be responsible for scoring.

        Returns:
            A model object
        """
        from sentence_transformers.cross_encoder import CrossEncoder

        return CrossEncoder(model_name=self.model_name)

    def _call(self, question: str, answer: str):
        """Runs the model to score the predictions.

        Args:
            question (str): The input text.
            answer (str): This can be the output from an LLM or the answer from a question-answer pair.

        Returns:
            Cross Answer relevancy score of the question and the answer.
        """
        scores = self.model.predict([[question, answer]])
        return softmax(scores[0])[2]
