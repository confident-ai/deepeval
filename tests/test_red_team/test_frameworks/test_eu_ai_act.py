import pytest
from deepteam.frameworks import EUAIAct
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.risk_category import RiskCategory
from deepteam.frameworks.eu_ai_act.risk_categories import (
    EU_AI_ACT_CATEGORIES,
)
from deepteam import red_team


EU_AI_ACT_DEFAULT_CATEGORIES = {
    "subliminal_manipulation",
    "exploitation_of_vulnerabilities",
    "social_scoring",
    "biometric_categorisation",
    "remote_biometric_id_live",
    "remote_biometric_id_post",
    "biometric_id",
    "critical_infrastructure",
    "education",
    "employment",
    "essential_services",
    "law_enforcement",
    "migration_border",
    "justice_democracy",
}


class TestEUAIAct:

    def test_eu_ai_act_init(self):
        """Test that EU AI Act framework can be instantiated."""
        framework = EUAIAct()
        assert framework is not None

    def test_eu_ai_act_name(self):
        """Test that EU AI Act framework has correct name."""
        framework = EUAIAct()
        assert (
            framework.name
            == framework.get_name()
            == "EU Artificial Intelligence Act"
        )

    def test_eu_ai_act_description(self):
        """Test that EU AI Act framework has a non-empty description."""
        framework = EUAIAct()
        assert isinstance(framework.description, str)
        assert len(framework.description) > 0

    def test_eu_ai_act_default_categories(self):
        """Test that all EU AI Act categories are included by default."""
        framework = EUAIAct()
        assert set(framework.categories) == EU_AI_ACT_DEFAULT_CATEGORIES

    def test_eu_ai_act_partial_categories(self):
        """Test that EU AI Act can be initialized with limited categories."""
        framework = EUAIAct(categories=["social_scoring", "employment"])
        assert set(framework.categories) == {
            "social_scoring",
            "employment",
        }
        assert len(framework.risk_categories) == 2

    def test_eu_ai_act_vulnerabilities_exist(self):
        """Test that EU AI Act framework defines vulnerabilities."""
        for risk_category in EU_AI_ACT_CATEGORIES:
            assert hasattr(risk_category, "vulnerabilities")
            assert len(risk_category.vulnerabilities) > 0

    def test_eu_ai_act_vulnerabilities_are_instances(self):
        """Test that all vulnerabilities are instances of BaseVulnerability."""
        for risk_category in EU_AI_ACT_CATEGORIES:
            for vuln in risk_category.vulnerabilities:
                assert isinstance(vuln, BaseVulnerability)

    def test_eu_ai_act_attacks_exist(self):
        """Test that EU AI Act framework defines attacks."""
        for risk_category in EU_AI_ACT_CATEGORIES:
            assert hasattr(risk_category, "attacks")
            assert len(risk_category.attacks) > 0

    def test_eu_ai_act_attacks_are_instances(self):
        """Test that all attacks are instances of BaseAttack."""
        for risk_category in EU_AI_ACT_CATEGORIES:
            for attack in risk_category.attacks:
                assert isinstance(attack, BaseAttack)

    def test_eu_ai_act_category_vulnerability_mapping(self):
        """Test that all categories map to vulnerabilities properly."""
        categories = EU_AI_ACT_CATEGORIES
        assert (
            set([category.name for category in categories])
            == EU_AI_ACT_DEFAULT_CATEGORIES
        )
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(
                isinstance(v, BaseVulnerability)
                for v in risk_category.vulnerabilities
            )

    def test_eu_ai_act_category_attack_mapping(self):
        """Test that all categories map to attacks properly."""
        categories = EU_AI_ACT_CATEGORIES
        assert (
            set([category.name for category in categories])
            == EU_AI_ACT_DEFAULT_CATEGORIES
        )
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(isinstance(a, BaseAttack) for a in risk_category.attacks)

    def test_eu_ai_act_vulnerability_names_present(self):
        """Test that key EU AI Act vulnerabilities are present."""
        vuln_names = []
        for risk_category in EU_AI_ACT_CATEGORIES:
            vuln_names.extend(
                [v.__class__.__name__ for v in risk_category.vulnerabilities]
            )
        expected_vulns = [
            "Bias",
            "Fairness",
            "Ethics",
            "PIILeakage",
            "PromptLeakage",
            "Misinformation",
            "Hallucination",
            "Robustness",
            "ExcessiveAgency",
            "IntellectualProperty",
            "ShellInjection",
            "SQLInjection",
            "SSRF",
            "ChildProtection",
            "Toxicity",
            "CustomVulnerability",
        ]
        for name in expected_vulns:
            assert (
                name in vuln_names
            ), f"Expected vulnerability {name} not found"

    def test_eu_ai_act_attack_names_present(self):
        """Test that key EU AI Act attacks are present."""
        attack_names = []
        for risk_category in EU_AI_ACT_CATEGORIES:
            attack_names.extend(
                [a.__class__.__name__ for a in risk_category.attacks]
            )
        expected_attacks = [
            "PromptInjection",
            "PromptProbing",
            "Roleplay",
            "GrayBox",
            "LinearJailbreaking",
            "TreeJailbreaking",
            "CrescendoJailbreaking",
            "SequentialJailbreak",
            "BadLikertJudge",
            "Multilingual",
            "Base64",
            "ROT13",
        ]
        for name in expected_attacks:
            assert name in attack_names, f"Expected attack {name} not found"

    def test_eu_ai_act_attack_weights_valid(self):
        """Test that all attacks have valid weights."""
        for risk_category in EU_AI_ACT_CATEGORIES:
            for attack in risk_category.attacks:
                assert hasattr(attack, "weight")
                assert isinstance(attack.weight, int)
                assert 1 <= attack.weight <= 3

    def test_eu_ai_act_framework_with_red_team(self):
        import random

        """Test EU AI Act integration with red_team pipeline."""

        def mock_model_callback(prompt: str, turns=None) -> str:
            return "This request violates internal policy."

        categories = list(EU_AI_ACT_DEFAULT_CATEGORIES)
        random_category = random.choice(categories)
        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=EUAIAct(categories=[random_category]),
            async_mode=False,
            ignore_errors=False,
        )
        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment is not None
