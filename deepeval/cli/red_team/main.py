import asyncio
import yaml
import typer
import importlib.util
import sys
import os
from typing import Callable, Awaitable, Optional
from dataclasses import dataclass

from . import config
from .model_callback import load_model

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
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
from deepteam.red_teamer.risk_assessment import RiskAssessment

app = typer.Typer(name="deepteam")

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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _show_version(value: bool):
    from deepteam._version import __version__

    if value:
        typer.echo(f"deepteam version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        help="Show the version and exit.",
        is_flag=True,
        callback=lambda v: _show_version(v),
        is_eager=True,
    )
):
    """
    DeepTeam CLI for red teaming LLMs.
    """
    pass


@app.command("login")
def login():
    typer.echo(
        f"This feature is currently in beta. For more details, please contact support@confident-ai.com"
    )
    raise typer.Exit()


@app.command("run")
def run(
    config_file: str,
    max_concurrent: int = typer.Option(
        None,
        "-c",
        "--max-concurrent",
        help="Maximum concurrent operations (overrides config)",
    ),
    attacks_per_vuln: int = typer.Option(
        None,
        "-a",
        "--attacks-per-vuln",
        help="Number of attacks per vulnerability type (overrides config)",
    ),
    output_folder: str = typer.Option(
        None,
        "-o",
        "--output-folder",
        help="Path to the output folder for saving risk assessment results (overrides config)",
    ),
):
    """Run a red teaming execution based on a YAML configuration"""
    cfg = _load_config(config_file)
    config.apply_env()

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


@app.command("scan")
def scan(
    path: str = typer.Argument(".", help="File or directory to scan."),
    diff: Optional[str] = typer.Option(
        None,
        "--diff",
        help="Scan only files changed between two git refs, e.g. 'main..HEAD'.",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "-c",
        "--config",
        help="Config YAML path (otherwise auto-discovered at the scan root).",
    ),
    output_format: str = typer.Option(
        "markdown",
        "-f",
        "--format",
        help="Output format: markdown | sarif | json.",
    ),
    min_severity: Optional[str] = typer.Option(
        None,
        "--min-severity",
        help="Only report findings at or above: low | medium | high | critical.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "-p",
        "--provider",
        help=(
            "Scan engine: codex | claude-code | cursor. Defaults from the API "
            "key that is set (OPENAI_API_KEY=codex, ANTHROPIC_API_KEY=claude-code, "
            "CURSOR_API_KEY=cursor), else deepteam's built-in scanner (OpenAI API)."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Model for the chosen provider. Overrides the config.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Write the report to a file instead of stdout.",
    ),
    fail_on_findings: bool = typer.Option(
        True,
        "--fail-on-findings/--no-fail-on-findings",
        help="Exit with code 1 if any findings remain (useful in CI).",
    ),
    comment: bool = typer.Option(
        False,
        "--comment",
        help="Post findings to Confident AI so deepteam[bot] comments on the PR (GitHub Actions only).",
    ),
    pr: Optional[int] = typer.Option(
        None,
        "--pr",
        help="Pull request number to comment on (manual/workflow_dispatch runs). Defaults to the PR from the GitHub Actions event.",
    ),
):
    """Scan source code for AI-security vulnerabilities."""
    from deepteam.code_scanner import (
        CodeScanner,
        build_engine,
        collect_changed_files,
        collect_files,
        filter_by_severity,
        load_config,
        post_pr_comments,
        resolve_provider,
        to_json,
        to_markdown,
        to_sarif,
    )

    formatters = {
        "markdown": to_markdown,
        "sarif": to_sarif,
        "json": to_json,
    }
    if output_format not in formatters:
        raise typer.BadParameter(
            f"format must be one of {', '.join(formatters)}"
        )

    config.apply_env()

    scan_root = path if os.path.isdir(path) else os.path.dirname(path) or "."
    cfg = load_config(path=config_file, directory=scan_root)

    resolved_provider = resolve_provider(provider or cfg.provider)
    model_name = model or cfg.model
    engine = build_engine(resolved_provider, model_name)
    # deepeval's judge needs a deepeval model; harness engines manage their own.
    eval_model = load_model(model_name) if engine is None else None
    typer.echo(
        f"Scanning with provider '{resolved_provider}'"
        + (f" (model: {model_name})" if model_name else ""),
        err=True,
    )

    if diff:
        base, _, head = diff.partition("..")
        chunks = collect_changed_files(
            path,
            base=base,
            head=head or None,
            include=cfg.include,
            exclude=cfg.exclude,
        )
    else:
        chunks = collect_files(path, include=cfg.include, exclude=cfg.exclude)

    typer.echo(f"Scanning {len(chunks)} code chunk(s)...", err=True)

    scanner = CodeScanner(
        model=eval_model,
        engine=engine,
        vulnerabilities=cfg.vulnerabilities,
        instruction=cfg.instruction,
        # Harness engines spawn an agent per batch, so keep concurrency modest.
        max_concurrent=4 if engine is not None else 10,
    )
    # Harness engines benefit from concurrent batches; run them async.
    if engine is not None:
        findings = asyncio.run(scanner.a_scan(chunks))
    else:
        findings = scanner.scan(chunks)
    findings = filter_by_severity(findings, min_severity or cfg.min_severity)

    rendered = formatters[output_format](findings)
    if output:
        with open(output, "w") as f:
            f.write(rendered)
        typer.echo(f"Wrote {len(findings)} finding(s) to {output}", err=True)
    else:
        typer.echo(rendered)

    if comment:
        post_pr_comments(findings, pr_number=pr)

    if fail_on_findings and findings:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
