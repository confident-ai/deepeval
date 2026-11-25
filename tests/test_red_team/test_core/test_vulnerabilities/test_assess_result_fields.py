"""
Regression tests for the attack method and risk category on the standalone
`vulnerability.assess()` path.

`LLMTestCase` is configured `extra="ignore"`, so a keyword that matches no field is
discarded without an error. The vulnerability modules used to pass `attackMethod` /
`riskCategory` — the spelling the red-team API payloads use — while `RTTestCase`
declares `attack_method` / `risk_category`, so both values vanished from every test
case returned by `assess()` / `a_assess()`.
"""

import ast
from pathlib import Path

from deepeval.red_team.test_case import RTTestCase
from deepeval.red_team.risks import getRiskCategory
from deepeval.red_team.vulnerabilities import Bias
from deepeval.red_team.vulnerabilities.bias import BiasType

PACKAGE_DIR = (
    Path(__file__).resolve().parents[4]
    / "deepeval"
    / "red_team"
    / "vulnerabilities"
)


class _StubMetric:
    def __init__(self):
        self.score = 0.0
        self.reason = "stubbed"

    def measure(self, test_case):
        return None


def _bias_with_one_simulated_attack() -> Bias:
    bias = Bias(types=["race"], async_mode=False)
    bias.simulate_attacks = lambda purpose=None: [
        RTTestCase(
            vulnerability="Bias",
            vulnerability_type=BiasType.RACE,
            input="x",
            attack_method="Base64",
        )
    ]
    bias._get_metric = lambda vuln_type: _StubMetric()
    return bias


def test_assess_returns_the_attack_method_and_risk_category():
    """
    The documented standalone path must carry what the simulator produced, otherwise
    `attack_method_results` grouping and anything a caller reads off the result is
    built from missing data.
    """
    results = _bias_with_one_simulated_attack().assess(
        model_callback=lambda prompt: "output"
    )
    cases = [case for group in results.values() for case in group]
    assert len(cases) == 1
    assert cases[0].attack_method == "Base64"
    assert cases[0].risk_category == getRiskCategory(BiasType.RACE).value


def _kwargs_by_module():
    """Keywords each vulnerability module passes to an RTTestCase(...) call."""
    found = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "RTTestCase"
            ):
                names |= {
                    keyword.arg for keyword in node.keywords if keyword.arg
                }
        if names:
            found[path.relative_to(PACKAGE_DIR).as_posix()] = names
    return found


def test_modules_only_pass_real_field_names():
    """
    Guards the class of mistake, not just this instance: any keyword that is not an
    `RTTestCase` field is silently dropped, so it has to fail here instead.
    """
    fields = set(RTTestCase.model_fields)
    not_fields = {
        module: sorted(names - fields)
        for module, names in _kwargs_by_module().items()
        if names - fields
    }
    assert not_fields == {}
