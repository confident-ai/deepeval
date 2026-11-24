import pytest
from deepteam.frameworks import OWASPTop10
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.owasp.risk_categories import OWASP_CATEGORIES
from deepteam.frameworks.risk_category import RiskCategory
from deepteam import red_team


class TestOWASP:

    def test_owasp_init(self):
        """Test that OWASP framework can be instantiated."""
        framework = OWASPTop10()
        assert framework is not None

    def test_owasp_name(self):
        """Test that OWASP framework has correct name."""
        framework = OWASPTop10()
        assert (
            framework.name
            == framework.get_name()
            == "OWASP Top 10 for LLMs 2025"
        )

    def test_owasp_description(self):
        """Test that OWASP framework has correct description."""
        framework = OWASPTop10()
        assert (
            framework.description
            == "A comprehensive list of the most critical security risks associated with LLM applications. The 2025 edition includes 10 critical risks covering prompt injection, sensitive information disclosure, supply chain vulnerabilities, data poisoning, output handling, excessive agency, system prompt leakage, vector/embedding weaknesses, misinformation, and unbounded consumption. Each risk is tested using realistic attack techniques and vulnerability assessments."
        )

    def test_owasp_default_categories(self):
        """Test that all 10 OWASP LLM categories are included by default."""
        framework = OWASPTop10()
        assert set(framework.categories) == {
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        }

    def test_owasp_partial_categories(self):
        """Test that framework can be created with limited categories."""
        framework = OWASPTop10(categories=["LLM_01", "LLM_02"])
        assert set(framework.categories) == {"LLM_01", "LLM_02"}

    def test_owasp_vulnerabilities_exist(self):
        """Test that vulnerabilities are defined and populated."""
        for risk_category in OWASP_CATEGORIES:
            assert hasattr(risk_category, "vulnerabilities")
            assert risk_category.vulnerabilities is not None
            assert len(risk_category.vulnerabilities) > 0

    def test_owasp_vulnerabilities_are_instances(self):
        """Test that all vulnerabilities are instances of BaseVulnerability."""
        for risk_category in OWASP_CATEGORIES:
            for vuln in risk_category.vulnerabilities:
                assert isinstance(vuln, BaseVulnerability)

    def test_owasp_attacks_exist(self):
        """Test that attacks are defined and populated."""
        for risk_category in OWASP_CATEGORIES:
            assert hasattr(risk_category, "attacks")
            assert risk_category.attacks is not None
            assert len(risk_category.attacks) > 0

    def test_owasp_attacks_are_instances(self):
        """Test that all attacks are instances of BaseAttack."""
        for risk_category in OWASP_CATEGORIES:
            for attack in risk_category.attacks:
                assert isinstance(attack, BaseAttack)

    def test_owasp_vulnerability_types_present(self):
        """Test that key vulnerabilities are present."""
        vuln_names = []
        for risk_category in OWASP_CATEGORIES:
            vuln_names.extend(
                [v.__class__.__name__ for v in risk_category.vulnerabilities]
            )
        expected_vulns = [
            "Bias",
            "Toxicity",
            "Misinformation",
            "IllegalActivity",
            "PromptLeakage",
            "PIILeakage",
            "ExcessiveAgency",
            "Robustness",
            "IntellectualProperty",
            "Competition",
            "GraphicContent",
            "PersonalSafety",
            "RBAC",
            "BOLA",
            "BFLA",
            "SSRF",
            "DebugAccess",
            "ShellInjection",
            "SQLInjection",
            "CustomVulnerability",
        ]
        for name in expected_vulns:
            assert name in vuln_names, f"Missing vulnerability {name}"

    def test_owasp_attack_types_present(self):
        """Test that key attacks are included."""
        attack_names = []
        for risk_category in OWASP_CATEGORIES:
            attack_names.extend(
                [a.__class__.__name__ for a in risk_category.attacks]
            )
        expected_attacks = [
            "PromptInjection",
            "PromptProbing",
            "Base64",
            "ROT13",
            "Leetspeak",
            "GrayBox",
            "Roleplay",
            "CrescendoJailbreaking",
            "LinearJailbreaking",
            "TreeJailbreaking",
        ]
        for attack in expected_attacks:
            assert attack in attack_names, f"Expected attack {attack} not found"

    def test_owasp_attack_weights_defined(self):
        """Test that all attacks have a valid weight."""
        for risk_category in OWASP_CATEGORIES:
            for attack in risk_category.attacks:
                assert hasattr(attack, "weight")
                assert isinstance(attack.weight, int)
                assert attack.weight >= 1

    def test_owasp_category_vulnerability_mapping(self):
        """Test that all categories map to vulnerabilities properly."""
        categories = OWASP_CATEGORIES
        assert set([category.name for category in categories]) == {
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(
                isinstance(v, BaseVulnerability)
                for v in risk_category.vulnerabilities
            )

    def test_owasp_category_attack_mapping(self):
        """Test that all categories map to attacks properly."""
        categories = OWASP_CATEGORIES
        assert set([category.name for category in categories]) == {
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(isinstance(a, BaseAttack) for a in risk_category.attacks)

    def test_owasp_framework_with_red_team(self):
        import random

        """Test that OWASP framework integrates correctly with red_team function."""

        # Simple mock model for testing
        def mock_model_callback(prompt: str, turns=None) -> str:
            return "I am a safe AI model and cannot perform unsafe actions."

        categories = [
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        ]
        random_category = random.choice(categories)

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=OWASPTop10(categories=[random_category]),
            async_mode=False,
            ignore_errors=False,
        )

        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment is not None
