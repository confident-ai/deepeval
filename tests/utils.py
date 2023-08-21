def assert_viable_score(score: float):
    assert score >= 0, f"Score {score} is less than 0"
    assert score <= 1, f"Score {score} is greater than 1"
