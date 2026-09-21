from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from deepeval.cli.dotenv_handler import DotenvHandler

# Written as bytes so the fixture is the same on every platform, whatever the
# ANSI codepage happens to be.
EXISTING_ENV = (
    "# comentário do usuário\n"
    'APP_GREETING="Ángel-gpt4"\n'
    "APP_THANKS=ありがとう\n"
    "OPENAI_API_KEY=sk-old\n"
).encode("utf-8")


def test_upsert_preserves_unrelated_non_ascii_values(tmp_path: Path):
    env_path = tmp_path / ".env.local"
    env_path.write_bytes(EXISTING_ENV)

    DotenvHandler(env_path).upsert({"OPENAI_API_KEY": "sk-new"})

    raw = env_path.read_bytes()
    assert "# comentário do usuário".encode("utf-8") in raw
    assert 'APP_GREETING="Ángel-gpt4"'.encode("utf-8") in raw
    assert "APP_THANKS=ありがとう".encode("utf-8") in raw

    values = dotenv_values(env_path)
    assert values["OPENAI_API_KEY"] == "sk-new"
    assert values["APP_GREETING"] == "Ángel-gpt4"
    assert values["APP_THANKS"] == "ありがとう"


def test_upsert_writes_non_ascii_values_readable_by_dotenv(tmp_path: Path):
    env_path = tmp_path / ".env.local"

    DotenvHandler(env_path).upsert(
        {
            "AZURE_DEPLOYMENT": "Ángel-gpt4",
            "APP_THANKS": "ありがとう",
            "APP_NOTE": "café au lait",
        }
    )

    values = dotenv_values(env_path)
    assert values["AZURE_DEPLOYMENT"] == "Ángel-gpt4"
    assert values["APP_THANKS"] == "ありがとう"
    assert values["APP_NOTE"] == "café au lait"


def test_upsert_roundtrips_non_ascii_across_two_calls(tmp_path: Path):
    env_path = tmp_path / ".env.local"
    handler = DotenvHandler(env_path)

    handler.upsert({"AZURE_DEPLOYMENT": "Ángel-gpt4"})
    handler.upsert({"OPENAI_API_KEY": "sk-new"})

    values = dotenv_values(env_path)
    assert values["AZURE_DEPLOYMENT"] == "Ángel-gpt4"
    assert values["OPENAI_API_KEY"] == "sk-new"


def test_unset_removes_only_its_key(tmp_path: Path):
    env_path = tmp_path / ".env.local"
    env_path.write_bytes(EXISTING_ENV)

    DotenvHandler(env_path).unset(["OPENAI_API_KEY"])

    raw = env_path.read_bytes()
    assert b"OPENAI_API_KEY" not in raw
    assert "# comentário do usuário".encode("utf-8") in raw
    assert 'APP_GREETING="Ángel-gpt4"'.encode("utf-8") in raw
    assert "APP_THANKS=ありがとう".encode("utf-8") in raw

    values = dotenv_values(env_path)
    assert "OPENAI_API_KEY" not in values
    assert values["APP_GREETING"] == "Ángel-gpt4"
    assert values["APP_THANKS"] == "ありがとう"
