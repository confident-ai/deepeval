import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

# Typer CLI entrypoint
from deepeval.cli.main import app

# Model + pricing table we want to get populated
from deepeval.models.llms.openai_model import GPTModel, model_pricing


SNAPSHOT = "gpt-4o-2024-08-06"
INPUT_PRICE = 2.5e-6
OUTPUT_PRICE = 1.0e-5


@pytest.fixture(autouse=True)
def _isolate_tmp_cwd(tmp_path, monkeypatch):
    # run everything from a clean temp dir
    monkeypatch.chdir(tmp_path)
    # point the JSON store to a file under this temp dir
    monkeypatch.setenv(
        "DEEPEVAL_KEY_FILE", str(tmp_path / ".deepeval" / ".deepeval")
    )
    # silence telemetry to reduce noise
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    # provide a dummy API key to satisfy existence
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    (tmp_path / ".deepeval").mkdir(parents=True, exist_ok=True)


def test_cli_set_openai_persists_and_enables_snapshot_pricing(tmp_path):
    runner = CliRunner()

    # run the CLI like using the same parameters that lep did
    result = runner.invoke(
        app,
        [
            "set-openai",
            "--model",
            SNAPSHOT,
            "--cost_per_input_token",
            str(INPUT_PRICE),
            "--cost_per_output_token",
            str(OUTPUT_PRICE),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    # confirm the JSON store was written in this CWD
    store_path = Path(".deepeval/.deepeval")
    assert store_path.exists(), "Expected .deepeval/.deepeval to be created"

    # assert that our environment variables are loaded
    data = json.loads(store_path.read_text())
    assert data.get("OPENAI_MODEL_NAME") == SNAPSHOT
    assert data.get("OPENAI_COST_PER_INPUT_TOKEN") == str(INPUT_PRICE)
    assert data.get("OPENAI_COST_PER_OUTPUT_TOKEN") == str(OUTPUT_PRICE)

    # now instantiate GPTModel WITHOUT passing costs; it must pull from store
    model = GPTModel()
    assert model.get_model_name() == SNAPSHOT

    # pricing table should have been populated for the snapshot
    assert (
        SNAPSHOT in model_pricing
    ), "Snapshot should be inserted into model_pricing"
    assert model_pricing[SNAPSHOT]["input"] == float(INPUT_PRICE)
    assert model_pricing[SNAPSHOT]["output"] == float(OUTPUT_PRICE)

    # and cost calculation shouldn't KeyError/ValueError
    _ = model.calculate_cost(1000, 2000)


# def test_cli_set_openai_with_dotenv_save_works_from_same_cwd(tmp_path):
#     runner = CliRunner()

#     # Save to dotenv in CWD to simulate your branch's `--save` flow
#     result = runner.invoke(
#         app,
#         [
#             "set-openai",
#             "--model", SNAPSHOT,
#             "--cost_per_input_token", str(INPUT_PRICE),
#             "--cost_per_output_token", str(OUTPUT_PRICE),
#             "--save", "dotenv:.env.local",
#         ],
#         catch_exceptions=False,
#     )
#     assert result.exit_code == 0, f"CLI failed:\n{result.output}"
#     assert Path(".env.local").exists(), "Expected .env.local to be created"

#     # Instantiation should still succeed (dotenv autoload at import time)
#     model = GPTModel()
#     assert model.get_model_name() == SNAPSHOT
#     assert SNAPSHOT in model_pricing
