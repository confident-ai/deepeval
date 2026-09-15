from typing import Optional

from pydantic import BaseModel

from deepeval.metrics.utils import verdict_from_json, verdicts_from_json


class Verdict(BaseModel):
    verdict: str
    reason: Optional[str] = None


def test_verdict_from_json_normalizes_verbose_verdict():
    verdict = verdict_from_json(
        {
            "verdict": " Yes, the assistant violated its assigned role.",
            "reason": "The response claims to be a doctor.",
        },
        Verdict,
    )

    assert verdict == Verdict(
        verdict="yes", reason="The response claims to be a doctor."
    )


def test_verdicts_from_json_excludes_unrecognized_verdicts():
    verdicts = verdicts_from_json(
        {
            "verdicts": [
                {"verdict": "No, the instruction was not followed."},
                {"verdict": "unclear"},
                {"verdict": "YES - a violation occurred"},
            ]
        },
        Verdict,
    )

    assert verdicts == [Verdict(verdict="no"), Verdict(verdict="yes")]
