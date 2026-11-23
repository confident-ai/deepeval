import pytest

from deepeval.models import DeepEvalBaseEmbeddingModel


class WordEmbedding(DeepEvalBaseEmbeddingModel):
    VOCAB = ["refund", "money", "back", "upgrade", "plan", "slow", "reply"]

    def load_model(self):
        return self

    def embed_text(self, text):
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        return [
            [1.0 if word in text.lower() else 0.0 for word in self.VOCAB]
            for text in texts
        ]

    async def a_embed_text(self, text):
        return self.embed_text(text)

    async def a_embed_texts(self, texts):
        return self.embed_texts(texts)

    def get_model_name(self):
        return "words"


@pytest.fixture(autouse=True)
def word_embedding(monkeypatch):
    monkeypatch.setattr(
        "deepeval.classifiers.classifier.config.initialize_embedding_model",
        lambda model=None: model or WordEmbedding(),
    )
