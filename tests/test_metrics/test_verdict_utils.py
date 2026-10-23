import json
from types import SimpleNamespace
from typing import Optional

import pytest

from pydantic import BaseModel

from deepeval.metrics.base_metric import YES_NO
from deepeval.metrics.utils import verdict_from_json
from deepeval.metrics.utils.qag import _verdicts_from_json


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
    verdicts = _verdicts_from_json(
        {
            "verdicts": [
                {"verdict": "No, the instruction was not followed."},
                {"verdict": "unclear"},
                {"verdict": "YES - a violation occurred"},
            ]
        },
        Verdict,
        YES_NO,
    )

    assert verdicts == [Verdict(verdict="no"), Verdict(verdict="yes")]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema_result", [False, True])
@pytest.mark.asyncio
async def test_list_generation_normalizes_or_filters_verdicts(
    asynchronous, schema_result
):
    from deepeval.metrics.prompt_alignment.schema import (
        PromptAlignmentVerdict,
        Verdicts,
    )
    from deepeval.metrics.utils import (
        generate_qag_verdicts,
        a_generate_qag_verdicts,
    )

    payload = {
        "verdicts": [
            {"verdict": "no" if schema_result else "No, not aligned."},
            {"verdict": "yes" if schema_result else "YES - aligned"},
        ]
    }
    if not schema_result:
        payload["verdicts"].append({"verdict": "unclear"})
    result = Verdicts(**payload) if schema_result else json.dumps(payload)

    class Judge:
        def generate_with_schema(self, *args, **kwargs):
            return result

        async def a_generate_with_schema(self, *args, **kwargs):
            return result

    metric = SimpleNamespace(model=Judge(), using_native_model=False)
    kwargs = dict(
        metric=metric,
        prompt="judge",
        verdict_cls=PromptAlignmentVerdict,
        verdicts_cls=Verdicts,
    )
    verdicts = (
        await a_generate_qag_verdicts(**kwargs)
        if asynchronous
        else generate_qag_verdicts(**kwargs)
    )
    assert [str(v.verdict) for v in verdicts] == ["no", "yes"]
