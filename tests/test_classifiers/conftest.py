import pytest

from tests.test_classifiers.fakes import WordEmbedding


@pytest.fixture(autouse=True)
def word_embedding(monkeypatch):
    monkeypatch.setattr(
        "deepeval.classifiers.classifier.config.initialize_embedding_model",
        lambda model=None: model or WordEmbedding(),
    )
