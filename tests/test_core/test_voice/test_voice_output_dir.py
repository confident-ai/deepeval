"""Where voice simulations write their audio.

`VoiceConfig.output_dir` has three states that are easy to confuse: left alone
it resolves through `DEEPEVAL_VOICE_FOLDER` to a default folder, set to a path
it writes there, and set to `None` it writes nothing. These pin down the
precedence between them, and that read-only mode outranks all three.
"""

import pytest

from deepeval.voice import VoiceConfig
from deepeval.voice.connectors.transports.callback import (
    CallbackVoiceConnector,
)
from deepeval.voice.output import DEFAULT_VOICE_FOLDER


@pytest.fixture
def connector():
    async def agent(audio):
        raise AssertionError("the agent is never called in these tests")

    return CallbackVoiceConnector(agent)


def test_defaults_to_a_namespaced_folder(connector):
    assert VoiceConfig(connector=connector).output_dir == (DEFAULT_VOICE_FOLDER)
    # Namespaced and hidden, so a run does not scatter audio through the cwd.
    assert DEFAULT_VOICE_FOLDER.startswith(".deepeval")


def test_env_var_replaces_the_default(connector, monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPEVAL_VOICE_FOLDER", str(tmp_path / "calls"))

    config = VoiceConfig(connector=connector)

    assert config.output_dir == str(tmp_path / "calls")


def test_env_var_expands_user_and_variables(connector, monkeypatch, tmp_path):
    monkeypatch.setenv("MY_CALLS", str(tmp_path / "expanded"))
    monkeypatch.setenv("DEEPEVAL_VOICE_FOLDER", "$MY_CALLS")

    config = VoiceConfig(connector=connector)

    assert config.output_dir == str(tmp_path / "expanded")


def test_an_explicit_folder_beats_the_env_var(connector, monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPEVAL_VOICE_FOLDER", str(tmp_path / "from-env"))

    config = VoiceConfig(
        connector=connector, output_dir=str(tmp_path / "from-code")
    )

    assert config.output_dir == str(tmp_path / "from-code")


def test_an_explicit_none_turns_writing_off_despite_the_env_var(
    connector, monkeypatch, tmp_path
):
    monkeypatch.setenv("DEEPEVAL_VOICE_FOLDER", str(tmp_path / "from-env"))

    assert VoiceConfig(connector=connector, output_dir=None).output_dir is None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default"),
        pytest.param({"output_dir": "somewhere"}, id="explicit-folder"),
    ],
)
def test_read_only_mode_writes_nothing(connector, monkeypatch, kwargs):
    """The one filesystem setting that has to outrank an explicit path."""
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")
    monkeypatch.setenv("DEEPEVAL_VOICE_FOLDER", "also-somewhere")

    assert VoiceConfig(connector=connector, **kwargs).output_dir is None
