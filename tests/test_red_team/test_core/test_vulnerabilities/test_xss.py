import pytest

from deepeval.red_team.vulnerabilities import XSS
from deepeval.red_team.vulnerabilities.xss import XSSType
from deepeval.red_team.vulnerabilities.xss import XSSTemplate
from deepeval.red_team.test_case import RTTestCase


class TestXSS:

    def test_xss_all_types(self):
        types = [
            "reflected_xss",
            "stored_xss",
            "dom_based_xss",
        ]
        xss = XSS(types=types)
        assert sorted(type.value for type in xss.types) == sorted(types)

    def test_xss_all_types_default(self):
        xss = XSS()
        assert sorted(type.value for type in xss.types) == sorted(
            type.value for type in XSSType
        )

    def test_xss_reflected_xss(self):
        types = ["reflected_xss"]
        xss = XSS(types=types)
        assert sorted(type.value for type in xss.types) == sorted(types)

    def test_xss_stored_xss(self):
        types = ["stored_xss"]
        xss = XSS(types=types)
        assert sorted(type.value for type in xss.types) == sorted(types)

    def test_xss_dom_based_xss(self):
        types = ["dom_based_xss"]
        xss = XSS(types=types)
        assert sorted(type.value for type in xss.types) == sorted(types)

    def test_xss_all_types_invalid(self):
        types = [
            "reflected_xss",
            "stored_xss",
            "dom_based_xss",
            "invalid",
        ]
        with pytest.raises(ValueError):
            XSS(types=types)

    def test_template_generates_attacks_for_each_type(self):
        # Pure (no model): every XSS sub-type must dispatch to a distinct,
        # non-empty baseline-attack prompt that asks for JSON output.
        prompts = {}
        for vuln_type in XSSType:
            prompt = XSSTemplate.generate_baseline_attacks(
                vuln_type, max_goldens=3, purpose="customer support chatbot"
            )
            assert isinstance(prompt, str) and prompt.strip()
            assert "JSON" in prompt
            assert "3" in prompt
            prompts[vuln_type] = prompt
        # Each sub-type produces its own prompt, not a shared one.
        assert len(set(prompts.values())) == len(list(XSSType))

    def test_simulate_attacks_returns_expected_cases(self):
        xss = XSS(types=["reflected_xss"])
        test_cases = xss.simulate_attacks(attacks_per_vulnerability_type=2)

        assert len(test_cases) == 2
        assert all(isinstance(tc, RTTestCase) for tc in test_cases)
        assert all(tc.vulnerability == "XSS" for tc in test_cases)
        assert all(
            tc.vulnerability_type == XSSType.REFLECTED_XSS for tc in test_cases
        )

    def test_assess_returns_results(self):
        xss = XSS(types=["reflected_xss"], async_mode=False)

        def dummy_model_callback(prompt):
            return prompt

        results = xss.assess(model_callback=dummy_model_callback)

        assert xss.is_vulnerable() is not None
        assert xss.simulated_attacks is not None and isinstance(
            xss.simulated_attacks, dict
        )
        assert xss.res is not None and isinstance(xss.res, dict)
        assert XSSType.REFLECTED_XSS in results
        assert len(results[XSSType.REFLECTED_XSS]) == 1
        test_case = results[XSSType.REFLECTED_XSS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")

    def test_get_metric_returns_XSS_metric(self):
        from deepeval.red_team._metrics import XSSMetric

        xss = XSS(async_mode=True, verbose_mode=True, evaluation_model="gpt-4o")
        metric = xss._get_metric(XSSType.REFLECTED_XSS)
        assert isinstance(metric, XSSMetric)
        assert metric.async_mode is True
        assert metric.verbose_mode is True

    @pytest.mark.asyncio
    async def test_a_assess_returns_async_results(self):
        xss = XSS(types=["reflected_xss"], async_mode=True)

        async def dummy_model_callback(prompt):
            return prompt

        results = await xss.a_assess(model_callback=dummy_model_callback)

        assert xss.is_vulnerable() is not None
        assert xss.simulated_attacks is not None and isinstance(
            xss.simulated_attacks, dict
        )
        assert xss.res is not None and isinstance(xss.res, dict)
        assert XSSType.REFLECTED_XSS in results
        assert len(results[XSSType.REFLECTED_XSS]) == 1
        test_case = results[XSSType.REFLECTED_XSS][0]
        assert hasattr(test_case, "score")
        assert hasattr(test_case, "reason")
