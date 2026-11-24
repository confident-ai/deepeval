from typing import List, Optional
from pydantic import BaseModel, Field

from deepeval.test_run.api import TurnApi
from deepteam.red_teamer.risk_assessment import (
    RiskAssessment,
    VulnerabilityTypeResult,
    AttackMethodResult,
)
from deepteam.test_case import RTTestCase, RTTurn


class VulnerabilityTypeResult(BaseModel):
    vulnerability: str
    vulnerability_type: str = Field(alias="vulnerabilityType")
    pass_rate: float = Field(alias="passRate")
    passing: int
    failing: int
    errored: int


class AttackMethodResult(BaseModel):
    attack_method: str = Field(alias="attackMethod")
    pass_rate: float = Field(alias="passRate")
    passing: int
    failing: int
    errored: int


class APIRTTurn(TurnApi):
    turn_level_attack: Optional[str] = Field(None, alias="turnLevelAttack")
    stopping_reason: Optional[str] = Field(None, alias="stoppingReason")
    stopping_category: Optional[str] = Field(None, alias="stoppingCategory")


class APIRTTestCase(BaseModel):
    input: str
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    retrieval_context: Optional[List[str]] = Field(
        None, alias="retrievalContext"
    )
    tools_called: Optional[List[str]] = Field(None, alias="toolsCalled")
    turns: Optional[List[APIRTTurn]] = Field(None)

    success: Optional[bool] = Field(None)
    score: Optional[float] = Field(None)
    reason: Optional[str] = Field(None)
    error: Optional[str] = Field(None)

    vulnerability: str = Field(alias="vulnerability")
    vulnerability_type: str = Field(alias="vulnerabilityType")
    attack_method: Optional[str] = Field(None, alias="attackMethod")
    risk_category: str = Field(alias="riskCategory")

    order: int


class APIRiskAssessment(BaseModel):
    vulnerability_results: List[VulnerabilityTypeResult] = Field(
        alias="vulnerabilityResults"
    )
    attack_results: List[AttackMethodResult] = Field(alias="attackResults")
    run_duration: float = Field(alias="runDuration")
    identifier: Optional[str] = Field(alias="identifier")
    assessment_cost: Optional[float] = Field(alias="assessmentCost")
    rt_framework_id: Optional[str] = Field(None, alias="rtFrameworkId")
    test_cases: List[APIRTTestCase] = Field(alias="testCases")


def map_turn_to_api(turn: RTTurn, order: int) -> APIRTTurn:
    """Map RTTurn to APIRTTurn."""
    return APIRTTurn(
        role=turn.role,
        content=turn.content,
        order=order,
        userId=turn.user_id,
        retrievalContext=turn.retrieval_context,
        toolsCalled=turn.tools_called,
        turnLevelAttack=turn.turn_level_attack,
        stoppingReason=turn.stopping_reason,
        stoppingCategory=turn.stopping_category,
    )


def map_test_case_to_api(test_case: RTTestCase, index: int) -> APIRTTestCase:
    """Map RTTestCase to APIRTTestCase."""
    turns = None
    if test_case.turns:
        turns = [
            map_turn_to_api(turn, idx)
            for idx, turn in enumerate(test_case.turns)
        ]

    if test_case.score is not None:
        success = test_case.score > 0
    else:
        success = None

    return APIRTTestCase(
        input=test_case.input,
        actualOutput=test_case.actual_output,
        retrievalContext=test_case.retrieval_context,
        toolsCalled=test_case.tools_called,
        turns=turns,
        success=success,
        score=test_case.score,
        reason=test_case.reason,
        error=test_case.error,
        vulnerability=test_case.vulnerability,
        vulnerabilityType=test_case.vulnerability_type.value,
        attackMethod=test_case.attack_method,
        riskCategory=test_case.risk_category,
        order=index,
    )


def map_vulnerability_type_result_to_api(
    result: VulnerabilityTypeResult,
) -> VulnerabilityTypeResult:
    """Map VulnerabilityTypeResult to API VulnerabilityTypeResult."""
    return VulnerabilityTypeResult(
        vulnerability=result.vulnerability,
        vulnerabilityType=result.vulnerability_type.value,
        passRate=result.pass_rate,
        passing=result.passing,
        failing=result.failing,
        errored=result.errored,
    )


def map_attack_method_result_to_api(
    result: AttackMethodResult,
) -> AttackMethodResult:
    """Map AttackMethodResult to API AttackMethodResult."""
    return AttackMethodResult(
        attackMethod=result.attack_method or "",
        passRate=result.pass_rate,
        passing=result.passing,
        failing=result.failing,
        errored=result.errored,
    )


def map_risk_assessment_to_api(
    risk_assessment: RiskAssessment,
    assessment_cost: Optional[float] = None,
    identifier: Optional[str] = None,
    rt_framework_id: Optional[str] = None,
) -> APIRiskAssessment:
    """
    Map RiskAssessment to APIRiskAssessment.

    Args:
        risk_assessment: The internal RiskAssessment object
        identifier: Optional[str] = None,
        assessment_cost: Optional[float] = None,
        rt_framework_id: id of the Confident AI framework this assessment ran,
            set only when the framework was pulled from Confident AI.

    Returns:
        APIRiskAssessment: The API-compatible risk assessment object
    """
    vulnerability_type_results = [
        map_vulnerability_type_result_to_api(result)
        for result in risk_assessment.overview.vulnerability_type_results
    ]

    attack_results = [
        map_attack_method_result_to_api(result)
        for result in risk_assessment.overview.attack_method_results
    ]

    test_cases = [
        map_test_case_to_api(test_case, index)
        for index, test_case in enumerate(risk_assessment.test_cases)
    ]

    return APIRiskAssessment(
        vulnerabilityResults=vulnerability_type_results,
        attackResults=attack_results,
        runDuration=risk_assessment.overview.run_duration,
        identifier=identifier,
        assessmentCost=assessment_cost,
        rtFrameworkId=rt_framework_id,
        testCases=test_cases,
    )
