from deepteam import red_team
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import *
from deepteam.attacks.multi_turn import *


async def model_callback(input: str, turns=None) -> str:
    return input


def test_red_teaming():
    vulnerabilities = [Bias(types=["race"])]

    attacks = [
        # Single-turn attacks
        AdversarialPoetry(),
        Base64(),
        GrayBox(),
        Leetspeak(),
        MathProblem(),
        Multilingual(),
        PromptInjection(),
        PromptProbing(),
        Roleplay(
            role="learned gentleman of natural philosophy",
            persona="1600s Shakespearean scholar",
        ),
        ROT13(),
        SystemOverride(),
        PermissionEscalation(),
        LinguisticConfusion(),
        InputBypass(),
        ContextPoisoning(),
        GoalRedirection(),
        # Multi-turn attacks
        CrescendoJailbreaking(),
        LinearJailbreaking(),
        TreeJailbreaking(),
        SequentialJailbreak(),
        BadLikertJudge(),
    ]

    risk_assessment = red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=1,
        max_concurrent=20,
        ignore_errors=False,
    )
    # assert len(risk_assessment.test_cases) > len(vulnerabilities) * len(attacks)
    assert risk_assessment is not None
    assert len(risk_assessment.test_cases) == 1
