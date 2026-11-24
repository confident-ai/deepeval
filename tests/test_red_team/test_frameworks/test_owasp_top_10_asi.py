import pytest
from deepteam.frameworks import OWASP_ASI_2026
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.owasp_top_10_agentic.risk_categories import (
    OWASP_ASI_CATEGORIES,
)
from deepteam.frameworks.risk_category import RiskCategory
from deepteam import red_team


class TestOWASP:

    def test_owasp_asi_2026_init(self):
        """Test that OWASP ASI 2026 framework can be instantiated."""
        framework = OWASP_ASI_2026()
        assert framework is not None

    def test_owasp_asi_2026_name(self):
        """Test that OWASP ASI 2026 framework has correct name."""
        framework = OWASP_ASI_2026()
        assert (
            framework.name
            == framework.get_name()
            == "OWASP Top 10 for Agentic Applications 2026"
        )

    def test_owasp_asi_2026_description(self):
        """Test that OWASP ASI 2026 framework has correct description."""
        framework = OWASP_ASI_2026()
        assert (
            framework.description
            == "A comprehensive list of the most critical security risks associated with agentic AI applications. The 2026 edition focuses on failures introduced by autonomy, tool usage, delegated trust, memory persistence, inter-agent communication, and emergent behavior. Each risk category is evaluated using realistic attack techniques and typed vulnerability assessments aligned with agent reasoning and behavior."
        )

    def test_owasp_asi_2026_default_categories(self):
        """Test that all 10 OWASP LLM categories are included by default."""
        framework = OWASP_ASI_2026()
        assert set(framework.categories) == {
            "ASI_01",
            "ASI_02",
            "ASI_03",
            "ASI_04",
            "ASI_05",
            "ASI_06",
            "ASI_07",
            "ASI_08",
            "ASI_09",
            "ASI_10",
        }

    def test_owasp_asi_2026_partial_categories(self):
        """Test that framework can be created with limited categories."""
        framework = OWASP_ASI_2026(categories=["ASI_01", "ASI_02"])
        assert set(framework.categories) == {"ASI_01", "ASI_02"}

    def test_owasp_asi_2026_vulnerabilities_exist(self):
        """Test that vulnerabilities are defined and populated."""
        for risk_category in OWASP_ASI_CATEGORIES:
            assert hasattr(risk_category, "vulnerabilities")
            assert risk_category.vulnerabilities is not None
            assert len(risk_category.vulnerabilities) > 0

    def test_owasp_asi_2026_vulnerabilities_are_instances(self):
        """Test that all vulnerabilities are instances of BaseVulnerability."""
        for risk_category in OWASP_ASI_CATEGORIES:
            for vuln in risk_category.vulnerabilities:
                assert isinstance(vuln, BaseVulnerability)

    def test_owasp_asi_2026_attacks_exist(self):
        """Test that attacks are defined and populated."""
        for risk_category in OWASP_ASI_CATEGORIES:
            assert hasattr(risk_category, "attacks")
            assert risk_category.attacks is not None
            assert len(risk_category.attacks) > 0

    def test_owasp_asi_2026_attacks_are_instances(self):
        """Test that all attacks are instances of BaseAttack."""
        for risk_category in OWASP_ASI_CATEGORIES:
            for attack in risk_category.attacks:
                assert isinstance(attack, BaseAttack)

    def test_owasp_asi_2026_vulnerability_types_present(self):
        """Test that key vulnerabilities are present."""
        vuln_names = []
        for risk_category in OWASP_ASI_CATEGORIES:
            vuln_names.extend(
                [v.__class__.__name__ for v in risk_category.vulnerabilities]
            )
        expected_vulns = [
            "Ethics",
            "Misinformation",
            "PromptLeakage",
            "BFLA",
            "BOLA",
            "RBAC",
            "DebugAccess",
            "ShellInjection",
            "SQLInjection",
            "IntellectualProperty",
            "GoalTheft",
            "RecursiveHijacking",
            "Robustness",
            "ExcessiveAgency",
            "IndirectInstruction",
            "ToolOrchestrationAbuse",
            "AgentIdentityAbuse",
            "ToolMetadataPoisoning",
            "UnexpectedCodeExecution",
            "InsecureInterAgentCommunication",
            "AutonomousAgentDrift",
        ]
        for name in expected_vulns:
            assert name in vuln_names, f"Missing vulnerability {name}"

    def test_owasp_asi_2026_attack_types_present(self):
        """Test that key attacks are included."""
        attack_names = []
        for risk_category in OWASP_ASI_CATEGORIES:
            attack_names.extend(
                [a.__class__.__name__ for a in risk_category.attacks]
            )
        expected_attacks = [
            "Base64",
            "PromptInjection",
            "PromptProbing",
            "Roleplay",
            "ROT13",
            "ContextPoisoning",
            "CrescendoJailbreaking",
            "LinearJailbreaking",
            "TreeJailbreaking",
        ]
        for attack in expected_attacks:
            assert attack in attack_names, f"Expected attack {attack} not found"

    def test_owasp_asi_2026_attack_weights_defined(self):
        """Test that all attacks have a valid weight."""
        for risk_category in OWASP_ASI_CATEGORIES:
            for attack in risk_category.attacks:
                assert hasattr(attack, "weight")
                assert isinstance(attack.weight, int)
                assert attack.weight >= 1

    def test_owasp_asi_2026_category_vulnerability_mapping(self):
        """Test that all categories map to vulnerabilities properly."""
        categories = OWASP_ASI_CATEGORIES
        assert set([category.name for category in categories]) == {
            "ASI_01",
            "ASI_02",
            "ASI_03",
            "ASI_04",
            "ASI_05",
            "ASI_06",
            "ASI_07",
            "ASI_08",
            "ASI_09",
            "ASI_10",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(
                isinstance(v, BaseVulnerability)
                for v in risk_category.vulnerabilities
            )

    def test_owasp_asi_2026_category_attack_mapping(self):
        """Test that all categories map to attacks properly."""
        categories = OWASP_ASI_CATEGORIES
        assert set([category.name for category in categories]) == {
            "ASI_01",
            "ASI_02",
            "ASI_03",
            "ASI_04",
            "ASI_05",
            "ASI_06",
            "ASI_07",
            "ASI_08",
            "ASI_09",
            "ASI_10",
        }
        for risk_category in categories:
            assert isinstance(risk_category, RiskCategory)
            assert all(isinstance(a, BaseAttack) for a in risk_category.attacks)

    def test_owasp_asi_2026_framework_with_red_team(self):
        import random

        """Test that OWASP ASI 2026 framework integrates correctly with red_team function."""

        # Simple mock model for testing
        def mock_model_callback(prompt: str, turns=None) -> str:
            return "I am a safe AI model and cannot perform unsafe actions."

        categories = [
            "ASI_01",
            "ASI_02",
            "ASI_03",
            "ASI_04",
            "ASI_05",
            "ASI_06",
            "ASI_07",
            "ASI_08",
            "ASI_09",
            "ASI_10",
        ]
        random_category = random.choice(categories)

        risk_assessment = red_team(
            model_callback=mock_model_callback,
            framework=OWASP_ASI_2026(categories=[random_category]),
            async_mode=False,
            ignore_errors=False,
        )

        assert isinstance(risk_assessment, RiskAssessment)
        assert risk_assessment is not None
