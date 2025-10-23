from deepeval.confident.api import is_confident, get_confident_api_key


def test_confident_boundary_off_in_core():
    assert get_confident_api_key() is None
    assert is_confident() is False
