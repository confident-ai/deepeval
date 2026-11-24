import logging
from typing import List, Optional

from deepteam.vulnerabilities import (
    BaseVulnerability,
    CustomVulnerability,
    Bias,
    Toxicity,
    Misinformation,
    IllegalActivity,
    PromptLeakage,
    PIILeakage,
    ExcessiveAgency,
    Robustness,
    IntellectualProperty,
    Competition,
    GraphicContent,
    PersonalSafety,
    RBAC,
    BOLA,
    BFLA,
    SSRF,
    DebugAccess,
    ShellInjection,
    SQLInjection,
)

from deepteam.vulnerabilities.constants import VULNERABILITY_CLASSES_MAP
from deepteam.metrics.types import EvaluationExample
from deepteam.frameworks.api import (
    APIAttackMethod,
    APIEvaluationExample,
    APIRiskCategory,
    APIVulnerability,
)
from deepteam.frameworks.risk_category import RiskCategory

from deepteam.attacks import BaseAttack
from deepteam.attacks.single_turn.constants import (
    SINGLE_TURN_ATTACK_CLASSES_MAP,
)
from deepteam.attacks.multi_turn.constants import MULTI_TURN_ATTACK_CLASSES_MAP
from deepteam.attacks.single_turn import (
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
)

from deepteam.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
)


def _get_owasp_vulnerabilities() -> List[BaseVulnerability]:
    """
    Get vulnerabilities based on OWASP Top 10 for LLMs 2025.

    Reference: docs/red-teaming-owasp-top-10-for-llms.mdx
    """
    return [
        # 1. Prompt Injection - Handled by attacks, not vulnerabilities
        # 2. Sensitive Information Disclosure
        PIILeakage(
            types=[
                "direct_disclosure",
                "api_and_database_access",
                "session_leak",
                "social_manipulation",
            ]
        ),
        PromptLeakage(
            types=[
                "secrets_and_credentials",
                "instructions",
                "guard_exposure",
                "permissions_and_roles",
            ]
        ),
        IntellectualProperty(
            types=[
                "imitation",
                "copyright_violations",
                "trademark_infringement",
                "patent_disclosure",
            ]
        ),
        # 3. Supply Chain - Not directly testable via attacks
        # 4. Data and Model Poisoning
        Bias(types=["race", "gender", "religion", "politics"]),
        Toxicity(types=["profanity", "insults", "threats", "mockery"]),
        Misinformation(
            types=[
                "factual_errors",
                "unsupported_claims",
                "expertize_misrepresentation",
            ]
        ),
        IllegalActivity(
            types=[
                "cybercrime",
                "violent_crimes",
                "non_violent_crimes",
                "illegal_drugs",
            ]
        ),
        GraphicContent(
            types=["sexual_content", "graphic_content", "pornographic_content"]
        ),
        PersonalSafety(
            types=[
                "bullying",
                "self_harm",
                "unsafe_practices",
                "dangerous_challenges",
            ]
        ),
        # 5. Improper Output Handling - Not directly testable via attacks
        # 6. Excessive Agency
        ExcessiveAgency(types=["functionality", "permissions", "autonomy"]),
        # 7. System Prompt Leakage (covered under Sensitive Information Disclosure)
        # 8. Vector and Embedding Weaknesses - Not directly testable via attacks
        # 9. Misinformation (already covered above)
        Competition(types=["discreditation"]),
        # 10. Unbounded Consumption - Not directly testable via attacks
        # Additional OWASP-related vulnerabilities
        RBAC(
            types=[
                "role_bypass",
                "privilege_escalation",
                "unauthorized_role_assumption",
            ]
        ),
        DebugAccess(
            types=[
                "debug_mode_bypass",
                "development_endpoint_access",
                "administrative_interface_exposure",
            ]
        ),
        ShellInjection(
            types=[
                "command_injection",
                "system_command_execution",
                "shell_escape_sequences",
            ]
        ),
        SQLInjection(
            types=[
                "blind_sql_injection",
                "union_based_injection",
                "error_based_injection",
            ]
        ),
        BFLA(
            types=[
                "privilege_escalation",
                "function_bypass",
                "authorization_bypass",
            ]
        ),
        BOLA(
            types=[
                "object_access_bypass",
                "cross_customer_access",
                "unauthorized_object_manipulation",
            ]
        ),
        SSRF(
            types=[
                "internal_service_access",
                "cloud_metadata_access",
                "port_scanning",
            ]
        ),
        Robustness(types=["hijacking", "input_overreliance"]),
    ]


def _get_owasp_attacks() -> List[BaseAttack]:
    """
    Get attacks relevant for OWASP Top 10 for LLMs testing.
    """
    return [
        # Core injection attacks
        PromptInjection(weight=3),
        # Encoding-based attacks for bypassing filters
        Base64(weight=2),
        ROT13(weight=2),
        Leetspeak(weight=2),
        # Social engineering attacks
        Roleplay(
            weight=2,
            role="learned gentleman of natural philosophy",
            persona="1600s Shakespearean scholar",
        ),
        PromptProbing(weight=2),
        # Multilingual and obfuscation attacks
        Multilingual(weight=2),
        MathProblem(weight=1),
        # Gray box attacks for known vulnerabilities
        GrayBox(weight=2),
        # Multi-turn sophisticated attacks
        CrescendoJailbreaking(weight=2),
        LinearJailbreaking(weight=2),
        TreeJailbreaking(weight=1),
        SequentialJailbreak(weight=2),
        BadLikertJudge(weight=1),
    ]


def _build_evaluation_examples(
    examples: Optional[List[APIEvaluationExample]],
) -> Optional[List[EvaluationExample]]:
    if not examples:
        return None

    return [
        EvaluationExample(
            input=example.input,
            actual_output=example.actual_output,
            score=example.score,
            reason=example.reason,
        )
        for example in examples
    ]


def _build_vulnerability(
    api_vulnerability: APIVulnerability, framework_name: str
) -> BaseVulnerability:
    evaluation_examples = _build_evaluation_examples(
        api_vulnerability.evaluation_examples
    )
    vulnerability_class = VULNERABILITY_CLASSES_MAP.get(api_vulnerability.name)

    if vulnerability_class is None:
        # A name deepteam doesn't know is either a custom vulnerability defined
        # on Confident AI (which carries the criteria needed to judge it) or a
        # vulnerability newer than the installed deepteam.
        if not api_vulnerability.criteria:
            raise ValueError(
                f"Vulnerability '{api_vulnerability.name}' in the '{framework_name}' framework "
                "is not supported by this version of deepteam. Upgrade with "
                "'pip install -U deepteam', or define it as a custom vulnerability on Confident AI."
            )
        return CustomVulnerability(
            name=api_vulnerability.name,
            criteria=api_vulnerability.criteria,
            types=api_vulnerability.types,
            evaluation_guidelines=api_vulnerability.evaluation_guidelines,
            evaluation_examples=evaluation_examples,
        )

    unsupported_types = [
        vulnerability_type
        for vulnerability_type in api_vulnerability.types
        if vulnerability_type not in vulnerability_class.ALLOWED_TYPES
    ]
    if unsupported_types:
        raise ValueError(
            f"Vulnerability '{api_vulnerability.name}' in the '{framework_name}' framework has "
            f"type(s) {', '.join(repr(t) for t in unsupported_types)} that are not supported by "
            "this version of deepteam. Upgrade with 'pip install -U deepteam'."
        )

    return vulnerability_class(
        types=api_vulnerability.types,
        evaluation_guidelines=api_vulnerability.evaluation_guidelines,
        evaluation_examples=evaluation_examples,
    )


def _build_attack(api_attack_method: APIAttackMethod) -> Optional[BaseAttack]:
    attack_classes_map = (
        MULTI_TURN_ATTACK_CLASSES_MAP
        if api_attack_method.multi_turn
        else SINGLE_TURN_ATTACK_CLASSES_MAP
    )
    attack_class = attack_classes_map.get(api_attack_method.name)

    if attack_class is None:
        logging.warning(
            f"Skipping attack method '{api_attack_method.name}': not supported by this "
            "version of deepteam. Upgrade with 'pip install -U deepteam' to run it."
        )
        return None

    parameters = api_attack_method.parameters or {}
    if parameters:
        try:
            return attack_class(**parameters)
        except Exception as e:
            logging.warning(
                f"Attack method '{api_attack_method.name}' rejected the parameters configured "
                f"on Confident AI ({e}). Falling back to defaults."
            )

    try:
        return attack_class()
    except Exception as e:
        # Some attacks take a required argument (e.g. Synthetic Context
        # Injection needs target_information), so there is no usable default to
        # fall back to. Skip it rather than fail the whole framework.
        logging.warning(
            f"Skipping attack method '{api_attack_method.name}': it cannot run without its "
            f"parameters ({e}). Configure them on Confident AI to include it."
        )
        return None


def build_risk_categories(
    api_risk_categories: List[APIRiskCategory], framework_name: str
) -> List[RiskCategory]:
    """
    Rebuild a Confident AI framework's risk categories as deepteam objects.

    Vulnerabilities and attack methods are resolved by name, which is how
    Confident AI stores them (its catalog is generated from deepteam's own
    class maps), so the local run tests the same threats as a cloud run.
    """
    risk_categories: List[RiskCategory] = []

    for api_risk_category in api_risk_categories:
        if not api_risk_category.vulnerabilities:
            logging.warning(
                f"Skipping risk category '{api_risk_category.name}': it has no vulnerability "
                "types selected, so there is nothing to test."
            )
            continue

        attacks = [
            attack
            for attack in (
                _build_attack(api_attack_method)
                for api_attack_method in api_risk_category.attack_methods
            )
            if attack is not None
        ]

        risk_categories.append(
            RiskCategory(
                name=api_risk_category.name,
                description=api_risk_category.description,
                vulnerabilities=[
                    _build_vulnerability(api_vulnerability, framework_name)
                    for api_vulnerability in api_risk_category.vulnerabilities
                ],
                # An empty list would have the simulator sample from no
                # attacks at all; None is what means "baseline attacks only".
                attacks=attacks or None,
            )
        )

    return risk_categories
