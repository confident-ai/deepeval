"""
Regression tests for #2904: `EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY` was
declared as a tuple `("LOCAL_EMBEDDING_API_KEY",)` instead of a plain string.

The enum member's `.value` is used as the JSON key in the key file
(`KeyFileHandler.data`), so a tuple key never matched the string keys stored
on disk: lookups like `should_use_ollama_embedding()` silently returned None.
"""

import os

from deepeval.key_handler import (
    EmbeddingKeyValues,
    KEY_FILE_HANDLER,
)
from deepeval.metrics.utils import should_use_ollama_embedding


def test_embedding_api_key_enum_value_is_plain_string():
    # Root cause: the member must carry the env-var name, not a tuple.
    assert isinstance(EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY.value, str)
    assert (
        EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY.value
        == "LOCAL_EMBEDDING_API_KEY"
    )


def test_ollama_embedding_selection_reads_stored_api_key(monkeypatch, tmp_path):
    import deepeval.key_handler as kh

    monkeypatch.setattr(kh, "HIDDEN_DIR", str(tmp_path))
    monkeypatch.setattr(kh, "KEY_FILE", "keys.json")
    os.makedirs(tmp_path, exist_ok=True)
    (tmp_path / "keys.json").write_text('{"LOCAL_EMBEDDING_API_KEY": "ollama"}')

    # Previously the tuple key never matched this string key, so this returned
    # False even though the API key was stored as "ollama".
    assert should_use_ollama_embedding() is True


def test_ollama_embedding_selection_defaults_to_false(monkeypatch, tmp_path):
    import deepeval.key_handler as kh

    monkeypatch.setattr(kh, "HIDDEN_DIR", str(tmp_path))
    monkeypatch.setattr(kh, "KEY_FILE", "keys.json")
    # No key file: the default (not Ollama) must be preserved.
    assert should_use_ollama_embedding() is False
