import os
import pytest

from deepeval.models.cohere_model import CohereModel


@pytest.mark.skip(reason="cohere is expensive")
def test_cohere_model():
    api_key = os.getenv("COHERE_API_KEY")
    model = CohereModel(api_key)

    prompt = "say hello in one word"
    res = model.generate(prompt)

    assert res is not None

if __name__ == '__main__':
    test_cohere_model()