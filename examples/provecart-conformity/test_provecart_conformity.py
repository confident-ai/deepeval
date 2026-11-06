"""ProveCart conformity practice set as a DeepEval/pytest suite.

Each practice instance becomes one test case: your agent decides,
the deterministic oracle (oracle.py — no LLM judge, no network)
scores the decision. Run with plain pytest, or with DeepEval:

    pytest test_provecart_conformity.py
    deepeval test run test_provecart_conformity.py

When DeepEval is installed, each case is also asserted through a
custom BaseMetric (ConformityMetric) so results land in DeepEval's
reporting with the oracle's reason as the metric reason.
"""
import json
import os

import pytest

from agent_under_test import decide
from oracle import load_practice_set, score_decision

_HERE = os.path.dirname(os.path.abspath(__file__))
_INSTANCES = load_practice_set(os.path.join(_HERE, "practice-set.json"))
_MATERIALS = {m["instance_id"]: m["materials"]
              for m in json.load(open(os.path.join(_HERE,
                                                   "materials.json")))}

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class ConformityMetric(BaseMetric):
        """Deterministic metric: 1.0 iff the decision was safe."""

        def __init__(self, instance):
            self.instance = instance
            self.threshold = 1.0

        def measure(self, test_case, *args, **kwargs):
            try:
                answer = json.loads(test_case.actual_output)
            except Exception:
                answer = None
            safe, reason = score_decision(self.instance, answer)
            self.score = 1.0 if safe else 0.0
            self.reason = reason
            self.success = safe
            return self.score

        async def a_measure(self, test_case, *args, **kwargs):
            return self.measure(test_case, *args, **kwargs)

        def is_successful(self):
            return bool(self.success)

        @property
        def __name__(self):
            return "Conformity"

    _HAS_DEEPEVAL = True
except Exception:
    _HAS_DEEPEVAL = False


@pytest.mark.parametrize("instance_id", sorted(_INSTANCES))
def test_safe_decision(instance_id):
    instance = _INSTANCES[instance_id]
    answer = decide(_MATERIALS[instance_id])
    safe, reason = score_decision(instance, answer)
    if _HAS_DEEPEVAL:
        from deepeval import assert_test
        case = LLMTestCase(
            input=instance["instruction"],
            actual_output=json.dumps(answer))
        assert_test(case, [ConformityMetric(instance)])
    assert safe, f"{instance['template_id']}: {reason}"
