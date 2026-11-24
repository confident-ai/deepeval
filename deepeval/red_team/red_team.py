from typing import List, Optional

from deepeval.models import DeepEvalBaseLLM
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.red_teamer import RedTeamer
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.frameworks.frameworks import RedTeamingFramework
from deepteam.red_teamer.cvss import Level


def red_team(
    model_callback: CallbackType,
    vulnerabilities: Optional[List[BaseVulnerability]] = None,
    attacks: Optional[List[BaseAttack]] = None,
    framework: Optional[RedTeamingFramework] = None,
    simulator_model: DeepEvalBaseLLM = "gpt-4o-mini",
    evaluation_model: DeepEvalBaseLLM = "gpt-4o-mini",
    attacks_per_vulnerability_type: int = 1,
    ignore_errors: bool = True,
    async_mode: bool = True,
    max_concurrent: int = 10,
    target_purpose: Optional[str] = None,
    attack_engine: Optional[AttackEngine] = None,
    identifier: Optional[str] = None,
    run_all_attacks: bool = False,
    exposure: Level = Level.MEDIUM,
):
    red_teamer = RedTeamer(
        async_mode=async_mode,
        max_concurrent=max_concurrent,
        target_purpose=target_purpose,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        attack_engine=attack_engine,
        exposure=exposure,
    )
    risk_assessment = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        framework=framework,
        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
        ignore_errors=ignore_errors,
        attack_engine=attack_engine,
        identifier=identifier,
        run_all_attacks=run_all_attacks,
    )
    return risk_assessment
