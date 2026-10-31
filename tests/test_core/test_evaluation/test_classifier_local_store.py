"""Classifier-only local persistence (#3538): TestResults, not a TestRun."""

import json
import re
from pathlib import Path

from deepeval.evaluate.local_store import write_classifier_results
from deepeval.evaluate.types import TestResult
from deepeval.test_run.api import Classification

CLASSIFIER_FILENAME_RE = re.compile(
    r"^classifier_run_\d{8}_\d{6}(?:_(\d+))?\.json$"
)


def _classifier_result() -> TestResult:
    return TestResult(
        name="abstains on gibberish",
        success=True,
        metrics_data=[],
        conversational=False,
        index=0,
        input="xyzzy",
        actual_output="qwerty",
        expected_output=None,
        classifications=[
            Classification(
                name="abstention",
                label="abstain",
                expectedLabel="abstain",
                success=True,
                reason="input is gibberish",
            )
        ],
    )


class TestWriteClassifierResults:
    def test_writes_timestamped_json_with_classifications(self, tmp_path: Path):
        path = write_classifier_results(tmp_path, [_classifier_result()])

        assert CLASSIFIER_FILENAME_RE.match(path.name) is not None
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["classifierRun"][0]["name"] == "abstains on gibberish"
        assert body["classifierRun"][0]["classifications"] == [
            {
                "name": "abstention",
                "label": "abstain",
                "expectedLabel": "abstain",
                "success": True,
                "reason": "input is gibberish",
            }
        ]
        assert "savedAt" in body

    def test_creates_missing_folder_and_empty_run_is_valid(
        self, tmp_path: Path
    ):
        target = tmp_path / "nested" / "results"
        path = write_classifier_results(target, [])

        assert path.exists()
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["classifierRun"] == []

    def test_same_second_collision_gets_suffix(self, tmp_path: Path):
        first = write_classifier_results(tmp_path, [_classifier_result()])
        second = write_classifier_results(tmp_path, [_classifier_result()])

        assert first != second
        assert CLASSIFIER_FILENAME_RE.match(second.name) is not None
