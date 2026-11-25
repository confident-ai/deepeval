import pytest
from deepeval.red_team.frameworks import ISO42001
from deepeval.red_team.red_teamer.risk_assessment import RiskAssessment
from deepeval.red_team.vulnerabilities import BaseVulnerability
from deepeval.red_team.attacks import BaseAttack
from deepeval.red_team.frameworks.iso_42001.risk_categories import (
    ISO_42001_CATEGORIES,
)
from deepeval.red_team.frameworks.risk_category import RiskCategory
from deepeval.red_team import red_team


ISO_42001_DEFAULT_CATEGORIES = {
    "A.6.2.4",
    "A.6.2.5",
    "A.6.2.6",
    "A.6.2.7",
    "A.6.2.8",
    "A.9.2",
    "A.9.3",
}


class TestISO42001:

    def test_iso_42001_init(self):
        """Test that ISO 42001 framework can be instantiated."""
        framework = ISO42001()
        assert framework is not None

    def test_iso_42001_name(self):
        """Test that ISO 42001 framework has correct name."""
        framework = ISO42001()
        assert (
            framework.name
            == framework.get_name()
            == "ISO/IEC 42001:2023 — AI Management Systems"
        )

    def test_iso_42001_description(self):
        """Test that ISO 42001 framework has a non-empty description."""
        framework = ISO42001()
        assert isinstance(framework.description, str)
        assert "ISO/IEC 42001:2023" in framework.description

    def test_iso_42001_default_categories(self):
        """Test that all default categories are included."""
        framework = ISO42001()
        assert set(framework.categories) == ISO_42001_DEFAULT_CATEGORIES

    def test_iso_42001_partial_categories(self):
        """Test that framework can be created with limited categories."""
        framework = ISO42001(categories=["A.6.2.4", "A.6.2.5"])
        assert set(framework.categories) == {"A.6.2.4", "A.6.2.5"}
        assert len(framework.risk_categories) == 2

    def test_iso_42001_invalid_category_raises(self):
        """Test that passing an unknown control id raises ValueError."""
        with pytest.raises(ValueError):
            ISO42001(categories=["A.99.9"])

    def test_iso_42001_vulnerabilities_exist(self):
        """Test that vulnerabilities are defined and populated."""
        for risk_category in ISO_42001_CATEGORIES:
            assert hasattr(risk_category, "vulnerabilities")
            assert risk_category.vulnerabilities is not None
            assert len(risk_category.vulnerabilities) > 0

    def test_iso_42001_vulnerabilities_are_instances(self):
        """Test that all vulnerabilities are instances of BaseVulnerability."""
        for risk_category in ISO_42001_CATEGORIES:
            for vuln in risk_category.vulnerabilities:
                assert isinstance(vuln, BaseVulnerability)

    def test_iso_42001_attacks_exist(self):
        """Test that attacks are defined and populated."""
        for risk_category in ISO_42001_CATEGORIES:
            assert hasattr(risk_category, "attacks")
            assert risk_category.attacks is not None
            assert len(risk_category.attacks) > 0

    def test_iso_42001_attacks_are_instances(self):
        """Test that all attacks are instances of BaseAttack."""
        for risk_category in ISO_42001_CATEGORIES:
            for attack in risk_category.attacks:
                assert isinstance(attack, BaseAttack)

    def test_iso_42001_attack_weights_valid(self):
        """Test that all attacks have a valid weight in the [1, 3] range."""
        for risk_category in ISO_42001_CATEGORIES:
            for attack in risk_category.attacks:
                assert hasattr(attack, "weight")
                assert isinstance(attack.weight, int)
                assert 1 <= attack.weight <= 3

    def test_iso_42001_category_vulnerability_mapping(self):
        """Test that all categories map to vulnerabilities properly."""
        categories = ISO_42001_CATEGORIES
        assert set([category.name for category in categories]) == {
            "a_6_2_4_fairness",
            "a_6_2_5_reliability_robustness",
            "a_6_2_6_security",
            "a_6_2_7_privacy",
            "a_6_2_8_transparency",
            "a_9_2_responsible_use",
            "a_9_3_human_oversight",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(
                isinstance(v, BaseVulnerability)
                for v in risk_category.vulnerabilities
            )

    def test_iso_42001_category_attack_mapping(self):
        """Test that all categories map to attacks properly."""
        categories = ISO_42001_CATEGORIES
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(isinstance(a, BaseAttack) for a in risk_category.attacks)

    def test_iso_42001_vulnerability_names_present(self):
        """Test that key ISO 42001 vulnerabilities are present across the framework."""
        vuln_names = []
        for risk_category in ISO_42001_CATEGORIES:
            vuln_names.extend(
                [v.__class__.__name__ for v in risk_category.vulnerabilities]
            )
        expected_vulns = [
            "Bias",
            "Toxicity",
            "Robustness",
            "Hallucination",
            "Misinformation",
            "PIILeakage",
            "PromptLeakage",
            "ShellInjection",
            "SQLInjection",
            "SSRF",
            "DebugAccess",
            "RBAC",
            "BFLA",
            "BOLA",
            "ExcessiveAgency",
            "IllegalActivity",
        ]
        for name in expected_vulns:
            assert (
                name in vuln_names
            ), f"Expected vulnerability {name} not found"

    def test_iso_42001_attack_names_present(self):
        """Test that key ISO 42001 attacks are present across the framework."""
        attack_names = []
        for risk_category in ISO_42001_CATEGORIES:
            attack_names.extend(
                [a.__class__.__name__ for a in risk_category.attacks]
            )
        expected_attacks = [
            "Roleplay",
            "Multilingual",
            "EmotionalManipulation",
            "LinearJailbreaking",
            "ContextPoisoning",
            "GoalRedirection",
            "SequentialJailbreak",
            "CrescendoJailbreaking",
            "TreeJailbreaking",
            "PromptInjection",
            "Base64",
            "AuthorityEscalation",
            "SystemOverride",
            "BadLikertJudge",
        ]
        for name in expected_attacks:
            assert name in attack_names, f"Expected attack {name} not found"

    def test_iso_42001_framework_with_red_team(self):
        import random

        """Test that ISO 42001 framework integrates correctly with red_team function."""

        # Simple mock model for testing
        def mock_model_callback(prompt: str, turns=None) -> str:
            return (
                "I'm a safe AI model. I cannot comply with unsafe instructions."
            )

        categories = ["A.6.2.4", "A.6.2.5", "A.6.2.6"]
        random_category = random.choice(categories)

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=ISO42001(categories=[random_category]),
            async_mode=False,
            ignore_errors=False,
        )

        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment is not None
