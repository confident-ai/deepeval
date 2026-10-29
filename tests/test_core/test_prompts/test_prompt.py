import types

import pytest

from tests.test_core.stubs import RecordingPortalockerLock
import deepeval.prompt.prompt as prompt_mod
from deepeval.prompt.api import PromptType, PromptInterpolationType
from tests.test_core.helpers import _make_fake_portalocker


@pytest.mark.parametrize(
    "cache_key_attr", ["VERSION_CACHE_KEY", "LABEL_CACHE_KEY"]
)
def test_write_to_cache_flushes_and_syncs(
    monkeypatch, tmp_path, cache_key_attr
):
    """
    Ensure Prompt._write_to_cache flushes and fsyncs after json.dump.

    This specifically protects against truncated JSON when multiple processes
    write to prompt cache on network filesystems.
    """

    fake_portalocker = _make_fake_portalocker()
    monkeypatch.setattr(
        prompt_mod, "portalocker", fake_portalocker, raising=False
    )

    # Use a temp directory for the cache path
    cache_path = tmp_path / "prompt_cache.json"
    monkeypatch.setattr(
        prompt_mod, "CACHE_FILE_NAME", str(cache_path), raising=False
    )
    monkeypatch.setattr(prompt_mod, "HIDDEN_DIR", str(tmp_path), raising=False)

    # Track fsync calls inside this module
    fsync_calls = []

    def fake_fsync(fd):
        fsync_calls.append(fd)

    monkeypatch.setattr(prompt_mod.os, "fsync", fake_fsync)

    # We don't need a real Prompt instance, just something with .alias
    dummy_self = types.SimpleNamespace(alias="my-alias")

    # Get the cache key constant (VERSION_CACHE_KEY or LABEL_CACHE_KEY)
    cache_key = getattr(prompt_mod, cache_key_attr)

    # Call the real method implementation, bound to our dummy object
    prompt_mod.Prompt._write_to_cache(
        dummy_self, cache_key=cache_key, hash="bab04ce"
    )

    # Assert file was flushed and synced
    f = RecordingPortalockerLock.last_file
    assert f is not None, "RecordingPortalockerLock did not capture a file"
    assert (
        f.flushed
    ), "Prompt._write_to_cache should call f.flush() after json.dump"
    assert (
        fsync_calls
    ), "Prompt._write_to_cache should call os.fsync(f.fileno())"
    assert fsync_calls[-1] == f.fileno()


def test_read_from_cache_branch_round_trip(monkeypatch, tmp_path):
    """A branch-pinned prompt written to the cache must be readable back.

    _write_to_cache stores branch pulls under the "branch" section only, but
    _read_from_cache gated the branch lookup on the presence of the unrelated
    "hash" section, so a valid cached branch prompt was a silent cache miss
    (and a spurious hard failure in the offline fallback path).
    """
    cache_path = tmp_path / "prompt_cache.json"
    monkeypatch.setattr(
        prompt_mod, "CACHE_FILE_NAME", str(cache_path), raising=False
    )
    monkeypatch.setattr(prompt_mod, "HIDDEN_DIR", str(tmp_path), raising=False)

    dummy_self = types.SimpleNamespace(alias="my-alias")

    # A branch pull writes a "branch" section and no "hash" section.
    prompt_mod.Prompt._write_to_cache(
        dummy_self,
        cache_key=prompt_mod.BRANCH_CACHE_KEY,
        hash="h1",
        branch="main",
        text_template="hello {name}",
        prompt_id="p1",
        type=PromptType.TEXT,
        interpolation_type=PromptInterpolationType.FSTRING,
    )

    cached = prompt_mod.Prompt._read_from_cache(
        dummy_self, alias="my-alias", branch="main"
    )

    assert (
        cached is not None
    ), "branch-pinned prompt in cache should be a hit, not a silent miss"
    assert cached.template == "hello {name}"
    assert cached.branch == "main"
