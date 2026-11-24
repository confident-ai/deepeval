import importlib.util
import os
import sys
from typing import Callable, Awaitable, Optional
from dataclasses import dataclass

from deepeval.cli.red_team.model_callback import load_model

from deepeval.red_team.red_teamer import RedTeamer
from deepeval.red_team.vulnerabilities import (
    # Data Privacy
    PIILeakage,
    PromptLeakage,
    # Responsible AI
    Bias,
    Toxicity,
    # Security
    BFLA,
    BOLA,
    RBAC,
    DebugAccess,
    ShellInjection,
    SQLInjection,
    SSRF,
    XSS,
    PathTraversal,
    # Safety
    IllegalActivity,
    GraphicContent,
    PersonalSafety,
    # Business
    Misinformation,
    IntellectualProperty,
    Competition,
    # Agentic
    GoalTheft,
    RecursiveHijacking,
    ExcessiveAgency,
    Robustness,
    # Custom
    CustomVulnerability,
)
from deepeval.red_team.attacks.single_turn import (
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
from deepeval.red_team.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
)
from deepeval.red_team.red_teamer.risk_assessment import RiskAssessment

VULN_CLASSES = [
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
    BFLA,
    BOLA,
    RBAC,
    DebugAccess,
    ShellInjection,
    SQLInjection,
    SSRF,
    XSS,
    PathTraversal,
    GoalTheft,
    RecursiveHijacking,
]
VULN_MAP = {cls.__name__: cls for cls in VULN_CLASSES}

ATTACK_CLASSES = [
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
]

ATTACK_MAP = {cls.__name__: cls for cls in ATTACK_CLASSES}


@dataclass
class ConfigRunResult:
    risk_assessment: RiskAssessment
    file_path: Optional[str] = None


def _build_vulnerability(cfg: dict, custom: bool):
    name = cfg.get("name")
    if not name:
        raise ValueError("Vulnerability entry missing 'name'")
    if custom:
        criteria = cfg.get("criteria")
        if not criteria:
            raise ValueError(
                "CustomVulnerability configuration must include a 'criteria' field that defines what should be evaluated."
            )
        return CustomVulnerability(
            name=name,
            criteria=criteria,
            types=cfg.get("types"),
            custom_prompt=cfg.get("prompt"),
        )

    cls = VULN_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown vulnerability: {name}")
    return cls(types=cfg.get("types"))


def _build_attack(cfg: dict):
    name = cfg.get("name")
    if not name:
        raise ValueError("Attack entry missing 'name'")
    cls = ATTACK_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown attack: {name}")
    kwargs = {}
    if "weight" in cfg:
        kwargs["weight"] = cfg["weight"]
    if "type" in cfg:
        kwargs["type"] = cfg["type"]
    if "persona" in cfg:
        kwargs["persona"] = cfg["persona"]
    if "category" in cfg:
        kwargs["category"] = cfg["category"]
    if "num_turns" in cfg:
        kwargs["num_turns"] = cfg["num_turns"]
    if "enable_refinement" in cfg:
        kwargs["enable_refinement"] = cfg["enable_refinement"]
    if "turn_level_attacks" in cfg:
        turn_level_attacks = cfg["turn_level_attacks"]
        attacks_objects = []
        for attack_name in turn_level_attacks:
            attack_cls = ATTACK_MAP.get(attack_name)
            if not attack_cls:
                raise ValueError(f"Unknown attack: {attack_name}")
            attacks_objects.append(attack_cls())
        kwargs["turn_level_attacks"] = attacks_objects
    obj = cls(**kwargs)
    return obj


def _load_callback_from_file(
    file_path: str, function_name: str
) -> Callable[[str], Awaitable[str]]:
    """Load a callback function from a Python file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target callback file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("target_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["target_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in {file_path}"
        )

    callback = getattr(module, function_name)
    if not callable(callback):
        raise TypeError(f"'{function_name}' in {file_path} is not callable")

    return callback


def _load_config(path: str):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(
    config_file: str,
    max_concurrent: Optional[int] = None,
    attacks_per_vuln: Optional[int] = None,
    output_folder: Optional[str] = None,
) -> ConfigRunResult:
    """Run a red teaming execution based on a YAML configuration."""
    cfg = _load_config(config_file)

    # Parse red teaming models (moved out of target section)
    models_cfg = cfg.get("models", {})

    simulator_model_spec = models_cfg.get("simulator", "gpt-3.5-turbo-0125")
    evaluation_model_spec = models_cfg.get("evaluation", "gpt-4o")

    simulator_model = None
    evaluation_model = None

    if not isinstance(simulator_model_spec, str):
        if "model" in simulator_model_spec:
            # Use model specification for simple cases
            _simulator_model_spec = simulator_model_spec["model"]
            simulator_model = load_model(_simulator_model_spec)

    if not isinstance(evaluation_model_spec, str):
        if "model" in evaluation_model_spec:
            # Use model specification for simple cases
            _evaluation_model_spec = evaluation_model_spec["model"]
            evaluation_model = load_model(_evaluation_model_spec)

    simulator_model = (
        load_model(simulator_model_spec)
        if not simulator_model
        else simulator_model
    )
    evaluation_model = (
        load_model(evaluation_model_spec)
        if not evaluation_model
        else evaluation_model
    )

    # Parse system configuration (renamed from options)
    system_config = cfg.get("system_config", {})
    final_max_concurrent = (
        max_concurrent
        if max_concurrent is not None
        else system_config.get("max_concurrent", 10)
    )
    final_attacks_per_vuln = (
        attacks_per_vuln
        if attacks_per_vuln is not None
        else system_config.get("attacks_per_vulnerability_type", 1)
    )
    final_output_folder = (
        output_folder
        if output_folder is not None
        else system_config.get("output_folder", None)
    )

    # Parse target configuration
    target_cfg = cfg.get("target", {})
    target_purpose = target_cfg.get("purpose", "")

    red_teamer = RedTeamer(
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        target_purpose=target_purpose,
        async_mode=system_config.get("run_async", True),
        max_concurrent=final_max_concurrent,
    )

    vulnerabilities_cfg = cfg.get("default_vulnerabilities", [])
    vulnerabilities = [
        _build_vulnerability(v, custom=False) for v in vulnerabilities_cfg
    ]

    custom_vulnerabilities_cfg = cfg.get("custom_vulnerabilities", [])
    vulnerabilities += [
        _build_vulnerability(v, custom=True) for v in custom_vulnerabilities_cfg
    ]

    attacks = [_build_attack(a) for a in cfg.get("attacks", [])]

    # Load target model callback - support both model specs and custom callbacks
    model_callback = None

    if "callback" in target_cfg:
        # Load custom callback from file
        callback_cfg = target_cfg["callback"]
        file_path = callback_cfg.get("file")
        function_name = callback_cfg.get("function", "model_callback")

        if not file_path:
            raise ValueError(
                "Target callback configuration missing 'file' field"
            )

        model_callback = _load_callback_from_file(file_path, function_name)

    elif "model" in target_cfg:
        # Use model specification for simple cases
        target_model_spec = target_cfg["model"]
        target_model = load_model(target_model_spec)

        if system_config.get("run_async", True):

            async def model_callback(input: str, turns=None) -> str:
                response = await target_model.a_generate(input)
                # Ensure we return a string, handle different response types
                if isinstance(response, tuple):
                    return str(response[0]) if response else "Empty response"
                return str(response)

        else:

            def model_callback(input: str, turns=None) -> str:
                response = target_model.generate(input)
                # Ensure we return a string, handle different response types
                if isinstance(response, tuple):
                    return str(response[0]) if response else "Empty response"
                return str(response)

    else:
        raise ValueError(
            "Target configuration must specify either 'model' or 'callback'"
        )

    risk = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=final_attacks_per_vuln,
        ignore_errors=system_config.get("ignore_errors", False),
    )
    result = ConfigRunResult(risk)

    # Save risk assessment if output folder is specified
    if final_output_folder is not None:
        file = red_teamer.risk_assessment.save(to=final_output_folder)
        result.file_path = file

    return result
