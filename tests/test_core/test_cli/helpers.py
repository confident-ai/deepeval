from __future__ import annotations

import re

from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typer.testing import CliRunner
from deepeval.key_handler import (
    KEY_FILE_HANDLER,
    KeyValues,
    ModelKeyValues,
    EmbeddingKeyValues,
)
from deepeval.cli.utils import USE_MODEL_KEYS

###########
# .env IO #
###########


def read_dotenv_as_dict(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def count_key_occurrences(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    prefix = f"{key}="
    return sum(
        1 for line in path.read_text().splitlines() if line.startswith(prefix)
    )


##############################
# enum <-> str normalization #
##############################


def key_name(k: Union[str, "Enum"]) -> str:
    # Works for ModelKeyValues / EmbeddingKeyValues / KeyValues or raw str
    return getattr(k, "value", k)


def normalize_keys(
    d: Mapping[Union[str, "Enum"], Optional[str]],
) -> Dict[str, str]:
    """Convert Enum keys to strings and drop None values."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        out[key_name(k)] = v
    return out


def normalize_key_list(keys: Iterable[Union[str, "Enum"]]) -> List[str]:
    return [key_name(k) for k in keys]


#################
# CLI execution #
#################


def build_args(
    command: str,
    *,
    positionals: Sequence[str] = (),
    options: Mapping[str, Optional[str]] = None,
    save_path: Optional[Path] = None,
    save_kind: Optional[str] = None,
) -> List[str]:
    """
    command="set-azure-openai"
    positionals=["llama3.2:3b"]                # for commands with a positional arg
    options={"--openai-endpoint": "..."}       # flags; None means boolean flag (omit value)
    save_path=Path("/tmp/.env")                # adds --save=dotenv:/tmp/.env
    """
    save_kind = "dotenv" if save_path and not save_kind else save_kind

    argv = [command]
    argv.extend(positionals)
    for flag, val in (options or {}).items():
        argv.append(flag)
        if val is not None:
            argv.append(str(val))
    if save_kind:
        argv.extend(
            [
                "--save",
                f"{save_kind}:{save_path}" if save_path else f"{save_kind}",
            ]
        )
    return argv


def invoke_ok(
    app, argv: Sequence[str], runner: Optional[CliRunner] = None
) -> Tuple[CliRunner, str]:
    """
    Runs the CLI and asserts exit_code == 0. Returns (runner, stdout).
    """
    runner = runner or CliRunner()
    result = runner.invoke(app, argv, catch_exceptions=False)
    assert (
        result.exit_code == 0
    ), f"command failed with exit_code:{result.exit_code} and output: {result.output}"
    return runner, result.output


def invoke_with_error(
    app,
    argv: Sequence[str],
    expected_exit_code: int = 1,
    expected_error: Optional[Union[str, Pattern[str]]] = None,
    runner: Optional[CliRunner] = None,
):
    """
    Run the CLI expecting a non zero exit. Optionally assert the error text.
    Returns (runner, stdout+stderr).
    """
    runner = runner or CliRunner()
    result = runner.invoke(app, argv, catch_exceptions=False)

    assert result.exit_code == expected_exit_code, (
        f"command was expected to fail with exit_code:{expected_exit_code}, "
        f"but got {result.exit_code} with output:\n{result.output}"
    )

    if expected_error is not None:
        if isinstance(expected_error, str):
            assert expected_error in result.output, (
                "command was expected to fail with message containing:\n"
                f"{expected_error}\n\nbut output was:\n{result.output}"
            )
        else:
            assert re.search(
                expected_error, result.output
            ), f"expected regex {expected_error.pattern!r} to match output:\n{result.output}"

    return runner, result.output


##############
# assertions #
##############


def assert_env_contains(
    path: Path, expected: Mapping[Union[str, "Enum"], str]
) -> None:
    env = read_dotenv_as_dict(path)
    for k, v in normalize_keys(expected).items():
        assert (
            env.get(k) == v
        ), f"Expected {k}={v!r} in {path}, got {env.get(k)!r}"


def assert_env_lacks(path: Path, keys: Iterable[Union[str, "Enum"]]) -> None:
    env = read_dotenv_as_dict(path)
    for k in normalize_key_list(keys):
        assert k not in env, f"Did not expect {k} in {path}, but found it"


def assert_no_dupes(path: Path, keys: Iterable[Union[str, "Enum"]]) -> None:
    for k in normalize_key_list(keys):
        assert (
            count_key_occurrences(path, k) == 1
        ), f"Key {k} duplicated in {path}"


def assert_deepeval_json_contains(
    expected: Mapping[
        Union[KeyValues, ModelKeyValues, EmbeddingKeyValues], str
    ],
) -> None:
    for k, v in expected.items():
        actual = KEY_FILE_HANDLER.fetch_data(k)
        assert (
            actual == v
        ), f"Expected {k}={v!r} in .deepeval JSON store, got {actual!r}"


def assert_deepeval_json_lacks(
    keys: Iterable[KeyValues | ModelKeyValues | EmbeddingKeyValues],
) -> None:
    for k in keys:
        actual = KEY_FILE_HANDLER.fetch_data(k)
        assert (
            actual == None
        ), f"Expected no key {k} in .deepeval JSON store, got {actual!r}"


def assert_env_model_switched_to(
    path: Path, selected_model: ModelKeyValues | EmbeddingKeyValues
):
    env = read_dotenv_as_dict(path)
    want_on, want_off = "1", "0"
    for k in USE_MODEL_KEYS:
        got = env.get(k.value)
        if k == selected_model:
            assert (
                got == want_on
            ), f"Expected {k}={want_on} in {path}, got {got!r}"
        else:
            assert (
                got == want_off
            ), f"Expected {k}={want_off} in {path}, got {got!r}"


def assert_deepeval_json_model_switched_to(
    selected_model: ModelKeyValues | EmbeddingKeyValues,
):
    for k in USE_MODEL_KEYS:
        actual = KEY_FILE_HANDLER.fetch_data(k)
        if k == selected_model:
            assert (
                actual == "YES"
            ), f"Expected {k}=YES in .deepeval/.deepeval, got {actual!r}"
        else:
            assert (
                actual == "NO"
            ), f"Expected {k}=NO in .deepeval/.deepeval, got {actual!r}"


def assert_model_switched_to(
    path: Path, selected_model: ModelKeyValues | EmbeddingKeyValues
):
    assert_env_model_switched_to(path, selected_model)
    assert_deepeval_json_model_switched_to(selected_model)
