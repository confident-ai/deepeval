from unittest.mock import MagicMock, patch

from deepeval.key_handler import EmbeddingKeyValues
from deepeval.metrics.utils import initialize_embedding_model


def _fetch_side_effect(active):
    def _fetch(key):
        if key == active:
            return "YES"
        return None

    return _fetch


@patch("deepeval.metrics.utils.GeminiEmbeddingModel")
@patch("deepeval.metrics.utils.KEY_FILE_HANDLER")
def test_initialize_selects_gemini_embedding(mock_kfh, mock_cls):
    mock_kfh.fetch_data.side_effect = _fetch_side_effect(
        EmbeddingKeyValues.USE_GEMINI_EMBEDDING
    )
    sentinel = MagicMock()
    mock_cls.return_value = sentinel

    result = initialize_embedding_model()

    mock_cls.assert_called_once_with()
    assert result is sentinel
