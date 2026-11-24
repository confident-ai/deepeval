from deepteam import red_team
from deepteam.vulnerabilities import *
from deepteam.attacks.single_turn import *
from deepteam.attacks.multi_turn import *


async def model_callback(input: str) -> str:
    return input


def test_red_teaming():

    custom_vulnerability = CustomVulnerability(
        criteria="Criteria",
        name="Name",
        # custom_prompt="Custom prompt",
        types=["type1", "type2"],
    )

    vulnerabilities = [
        Bias(),
        Toxicity(),
        Misinformation(),
        IllegalActivity(),
        PromptLeakage(),
        PIILeakage(),
        BFLA(),
        BOLA(),
        RBAC(),
        DebugAccess(),
        ShellInjection(),
        SQLInjection(),
        SSRF(),
        IntellectualProperty(),
        Competition(),
        GraphicContent(),
        PersonalSafety(),
        # Agentic vulnerabilities
        GoalTheft(),
        RecursiveHijacking(),
        Robustness(),
        ExcessiveAgency(),
        custom_vulnerability,
    ]

    attacks = [
        # Single-turn attacks
        Base64(),
        # GrayBox(),
        # Leetspeak(),
        # MathProblem(),
        # Multilingual(),
        # PromptInjection(),
        # PromptProbing(),
        # Roleplay(),
        # ROT13(),
        # SystemOverride(),
        # PermissionEscalation(),
        # LinguisticConfusion(),
        # InputBypass(),
        # ContextPoisoning(),
        # GoalRedirection(),
        # # Multi-turn attacks
        # CrescendoJailbreaking(),
        # LinearJailbreaking(),
        # TreeJailbreaking(),
        # SequentialJailbreak(),
        # BadLikertJudge(),
    ]

    risk_assessment = red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=1,
        max_concurrent=20,
        ignore_errors=False,
    )
    assert risk_assessment is not None
    assert len(risk_assessment.test_cases) > len(vulnerabilities) * len(attacks)
