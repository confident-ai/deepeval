import pytest
from deepteam.frameworks import NIST
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.nist.risk_categories import NIST_CATEGORIES
from deepteam.frameworks.risk_category import RiskCategory
from deepteam import red_team


class TestNIST:

    def test_nist_init(self):
        """Test that NIST framework can be instantiated."""
        framework = NIST()
        assert framework is not None

    def test_nist_name(self):
        """Test that NIST framework has correct name."""
        framework = NIST()
        assert (
            framework.name
            == framework.get_name()
            == "NIST AI Risk Management Framework (AI RMF)"
        )

    def test_nist_description(self):
        """Test that NIST framework has correct description."""
        framework = NIST()
        assert (
            framework.description
            == "A structured methodology from NIST for identifying, evaluating, and mitigating risks in AI systems. Confident AI's implementation focuses on the Measure function - testing, evaluation, and assurance of AI behavior and risk controls across four measurement categories (M.1-M.4) covering risk metrics, trustworthiness evaluation, risk tracking, and impact assessment."
        )

    def test_nist_default_categories(self):
        """Test that all four measure categories are included by default."""
        framework = NIST()
        assert set(framework.categories) == {
            "measure_1",
            "measure_2",
            "measure_3",
            "measure_4",
        }

    def test_nist_partial_categories(self):
        """Test that framework can be created with limited categories."""
        framework = NIST(categories=["measure_1", "measure_2"])
        assert set(framework.categories) == {"measure_1", "measure_2"}

    def test_nist_vulnerabilities_exist(self):
        """Test that vulnerabilities are defined and populated."""
        for risk_category in NIST_CATEGORIES:
            assert hasattr(risk_category, "vulnerabilities")
            assert risk_category.vulnerabilities is not None
            assert len(risk_category.vulnerabilities) > 0

    def test_nist_vulnerabilities_are_instances(self):
        """Test that all vulnerabilities are instances of BaseVulnerability."""
        for risk_category in NIST_CATEGORIES:
            for vuln in risk_category.vulnerabilities:
                assert isinstance(vuln, BaseVulnerability)

    def test_nist_attacks_exist(self):
        """Test that attacks are defined and populated."""
        for risk_category in NIST_CATEGORIES:
            assert hasattr(risk_category, "attacks")
            assert risk_category.attacks is not None
            assert len(risk_category.attacks) > 0

    def test_nist_attacks_are_instances(self):
        """Test that all attacks are instances of BaseAttack."""
        for risk_category in NIST_CATEGORIES:
            for attack in risk_category.attacks:
                assert isinstance(attack, BaseAttack)

    def test_nist_vulnerability_types_present(self):
        """Test that key vulnerabilities are present."""
        vuln_names = []
        for risk_category in NIST_CATEGORIES:
            vuln_names.extend(
                [v.__class__.__name__ for v in risk_category.vulnerabilities]
            )
        expected_vulns = [
            "IntellectualProperty",
            "RBAC",
            "Bias",
            "Fairness",
            "Ethics",
            "Toxicity",
            "Misinformation",
            "PIILeakage",
            "PromptLeakage",
            "IllegalActivity",
            "Robustness",
            "ExcessiveAgency",
            "ShellInjection",
            "SQLInjection",
            "SSRF",
            "GraphicContent",
            "PersonalSafety",
            "ChildProtection",
            "BOLA",
            "BFLA",
            "Competition",
        ]
        for name in expected_vulns:
            assert name in vuln_names, f"Missing vulnerability {name}"

    def test_nist_attack_types_present(self):
        """Test that key attacks are included."""
        attack_names = []
        for risk_category in NIST_CATEGORIES:
            attack_names.extend(
                [a.__class__.__name__ for a in risk_category.attacks]
            )
        expected_attacks = [
            "PromptInjection",
            "PromptProbing",
            "Base64",
            "ROT13",
            "Leetspeak",
            "Multilingual",
            "GrayBox",
            "Roleplay",
            "CrescendoJailbreaking",
            "LinearJailbreaking",
            "TreeJailbreaking",
            "SequentialJailbreak",
            "BadLikertJudge",
        ]
        for attack in expected_attacks:
            assert attack in attack_names, f"Expected attack {attack} not found"

    def test_nist_attack_weights_defined(self):
        """Test that all attacks have a valid weight."""
        for risk_category in NIST_CATEGORIES:
            for attack in risk_category.attacks:
                assert hasattr(attack, "weight")
                assert isinstance(attack.weight, int)
                assert attack.weight >= 1

    def test_nist_category_vulnerability_mapping(self):
        """Test that all categories map to vulnerabilities properly."""
        categories = NIST_CATEGORIES
        assert set([category.name for category in categories]) == {
            "measure_1",
            "measure_2",
            "measure_3",
            "measure_4",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(
                isinstance(v, BaseVulnerability)
                for v in risk_category.vulnerabilities
            )

    def test_nist_category_attack_mapping(self):
        """Test that all categories map to attacks properly."""
        categories = NIST_CATEGORIES
        assert set([category.name for category in categories]) == {
            "measure_1",
            "measure_2",
            "measure_3",
            "measure_4",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(isinstance(v, BaseAttack) for v in risk_category.attacks)

    def test_nist_framework_with_red_team(self):
        import random

        """Test that NIST framework integrates correctly with red_team function."""

        # Simple mock model for testing
        def mock_model_callback(prompt: str, turns=None) -> str:
            return (
                "I'm a safe AI model. I cannot comply with unsafe instructions."
            )

        categories = ["measure_1", "measure_2", "measure_3", "measure_4"]
        random_category = random.choice(
            categories
        )  # Randomizing to not test on too many attacks or vulnerabilities

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=NIST(categories=[random_category]),
            async_mode=False,
            ignore_errors=False,
        )

        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment is not None
