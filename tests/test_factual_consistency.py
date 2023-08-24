import pytest
from deepeval.metrics.factual_consistency import assert_factual_consistency


@pytest.mark.asyncio
async def test_factual_consistency():
    with pytest.raises(AssertionError):
        assert_factual_consistency(
            "Sarah spent the evening at the library, engrossed in a book.",
            "After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond.",
        )

    assert_factual_consistency(
        "Sarah went out for a walk in the park.",
        "After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond.",
    )
