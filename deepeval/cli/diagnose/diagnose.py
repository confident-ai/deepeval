"""`deepeval diagnose` Typer command.

Read-only report of the effective DeepEval configuration: which evaluation
models would be used, where each setting was loaded from (process env,
dotenv file, or the legacy JSON keystore), Confident AI login status,
data region and API endpoint.

This intentionally reuses the exact resolvers the rest of deepeval uses at
runtime (`get_settings`, `initialize_model`, `initialize_embedding_model`,
`get_base_api_url`, ...) so what it prints is what an eval run would
actually do — not a reimplementation that can drift.
"""

from __future__ import annotations

import importlib.metadata
import json as json_lib
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import typer
from pydantic import SecretStr
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table

from deepeval.config.settings import (
    dotenv_search_paths,
    get_settings,
    resolve_env_dir,
)
from deepeval.config.utils import parse_bool, read_dotenv_file
from deepeval.constants import HIDDEN_DIR, KEY_FILE
from deepeval.key_handler import (
    EmbeddingKeyValues,
    KeyValues,
    ModelKeyValues,
)

# Friendly provider labels for the model classes `initialize_model` /
# `initialize_embedding_model` can return (keyed by class name so we don't
# have to import every model class here).
_PROVIDER_BY_CLASS = {
    "GPTModel": "OpenAI",
    "AzureOpenAIModel": "Azure OpenAI",
    "OllamaModel": "Ollama",
    "LocalModel": "Local model",
    "GeminiModel": "Google Gemini",
    "AnthropicModel": "Anthropic",
    "AmazonBedrockModel": "AWS Bedrock",
    "LiteLLMModel": "LiteLLM",
    "KimiModel": "Moonshot (Kimi)",
    "GrokModel": "Grok",
    "DeepSeekModel": "DeepSeek",
    "OpenRouterModel": "OpenRouter",
    "PortkeyModel": "Portkey",
    "OpenAIEmbeddingModel": "OpenAI",
    "AzureOpenAIEmbeddingModel": "Azure OpenAI",
    "OllamaEmbeddingModel": "Ollama",
    "LocalEmbeddingModel": "Local model",
}

# Settings fields worth showing in the "configured settings" table when set.
_RELEVANT_MARKERS = (
    "CONFIDENT_",
    "USE_",
    "_API_KEY",
    "_MODEL_NAME",
    "_DEPLOYMENT_NAME",
    "_BASE_URL",
    "_ENDPOINT",
    "TEMPERATURE",
    "DEEPEVAL_DEFAULT_SAVE",
    "DEEPEVAL_RESULTS_FOLDER",
)


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return "*" * 8
    return "*" * 8 + value[-6:]


def _deepeval_version() -> str:
    try:
        return importlib.metadata.version("deepeval")
    except importlib.metadata.PackageNotFoundError:
        try:
            from deepeval import __version__  # type: ignore

            return __version__
        except ImportError:
            return "unknown"


def _dotenv_paths() -> List[Path]:
    """Dotenv files in precedence order (highest first). Same list
    `autoload_dotenv()` loads — `dotenv_search_paths()` is shared."""
    return list(reversed(dotenv_search_paths()))


def _legacy_keystore() -> Tuple[Path, Dict[str, Any]]:
    path = Path(HIDDEN_DIR) / KEY_FILE
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json_lib.load(f)
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json_lib.JSONDecodeError):
        data = {}
    return path, data


def _env_key_to_json_key() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for enum in (KeyValues, ModelKeyValues, EmbeddingKeyValues):
        for member in enum:
            mapping[member.name] = member.value
    return mapping


def _resolve_source(
    env_key: str,
    dotenv_values: List[Tuple[Path, Dict[str, str]]],
    legacy_data: Dict[str, Any],
    json_key_map: Dict[str, str],
) -> Optional[str]:
    """Best-effort provenance for a settings field.

    Mirrors runtime precedence: process env > dotenv (.env.local >
    .env.{APP_ENV} > .env) > legacy JSON keystore. Dotenv and keystore
    values are merged into os.environ at import time, so we attribute a
    value to a file when the file defines it with the same value.
    """
    raw = os.environ.get(env_key)

    for path, values in dotenv_values:
        if env_key in values and (raw is None or values[env_key] == raw):
            return path.name

    json_key = json_key_map.get(env_key)
    if json_key is not None:
        legacy_val = legacy_data.get(json_key)
        if legacy_val is not None and (raw is None or str(legacy_val) == raw):
            return f"{HIDDEN_DIR}/{KEY_FILE} (JSON keystore)"

    if raw is not None:
        return "process environment"
    return None  # built-in default / not set


def _display_value(value: Any) -> str:
    if isinstance(value, SecretStr):
        return _mask_secret(value.get_secret_value())
    return str(value)


def resolve_setting_source(env_key: str) -> Optional[str]:
    """Best-effort provenance for a single settings field (see
    `_resolve_source`). Used by `deepeval logout` to detect keys that
    resolve from the shell environment, which the CLI cannot unset."""
    dotenv_values = [
        (p, read_dotenv_file(p)) for p in _dotenv_paths() if p.is_file()
    ]
    _, legacy_data = _legacy_keystore()
    return _resolve_source(
        env_key, dotenv_values, legacy_data, _env_key_to_json_key()
    )


_REGION_SOURCE_LABELS = {
    "custom_base_url": "custom CONFIDENT_BASE_URL overrides region",
    "explicit_region": "explicitly set",
    "api_key_prefix": "inferred from API key prefix",
    "default": "default",
}

_OTEL_HOST_REGION = {
    "otel.confident-ai.com": "US",
    "eu.otel.confident-ai.com": "EU",
    "au.otel.confident-ai.com": "AU",
}
_OTEL_URL_BY_REGION = {
    region: f"https://{host}" for host, region in _OTEL_HOST_REGION.items()
}


def _region_warnings(
    region: Optional[str],
    region_source: str,
    api_key: Optional[str],
    otel_url: Optional[str],
) -> List[str]:
    """Flag endpoints that look like they route to the wrong data region."""
    from deepeval.confident.api import _infer_region_from_api_key

    if not region:
        # Custom CONFIDENT_BASE_URL: user is off the regional grid on purpose.
        return []

    warnings: List[str] = []

    key_region = _infer_region_from_api_key(api_key)
    if key_region and key_region != region:
        article = "an" if key_region in ("EU", "AU") else "a"
        warnings.append(
            f"Your API key prefix looks like {article} {key_region} key, but the "
            f"resolved data region is {region} ({region_source}). API calls "
            f"will go to the {region} endpoint — if that's wrong, run "
            f"`deepeval set-confident-region {key_region}`."
        )

    if otel_url:
        otel_host = urlsplit(otel_url).hostname
        otel_region = _OTEL_HOST_REGION.get(otel_host or "")
        if otel_region and otel_region != region:
            warnings.append(
                f"OTEL endpoint is the {otel_region} host but your data "
                f"region is {region}. OTLP traces would land in the wrong "
                f"region — set CONFIDENT_OTEL_URL="
                f"{_OTEL_URL_BY_REGION[region]}."
            )

    return warnings


def _confident_section() -> Dict[str, Any]:
    from deepeval.confident.api import (
        get_confident_api_key,
        is_confident,
        resolve_backend,
    )

    settings = get_settings()
    api_key = get_confident_api_key()

    # The same resolution every API call goes through — url, region and
    # provenance come from one place, so this cannot drift from runtime.
    backend = resolve_backend()
    region_source = _REGION_SOURCE_LABELS.get(backend.source, backend.source)

    # OTLP exporters read this setting directly (no region inference);
    # spans are posted to <url>/v1/traces.
    otel_url = (
        str(settings.CONFIDENT_OTEL_URL)
        if settings.CONFIDENT_OTEL_URL
        else None
    )

    return {
        "logged_in": is_confident(),
        "api_key": _mask_secret(api_key),
        "region": backend.region,
        "region_source": region_source,
        "api_base_url": backend.base_url,
        "otel_url": otel_url,
        "warnings": _region_warnings(
            backend.region, region_source, api_key, otel_url
        ),
    }


def _models_section() -> Dict[str, Any]:
    # Heavy import (pulls in the model providers); defer so `--help` stays fast.
    from deepeval.metrics.utils import (
        initialize_embedding_model,
        initialize_model,
    )

    result: Dict[str, Any] = {
        # These are the *global defaults*: what `initialize_model()` /
        # `initialize_embedding_model()` return when a consuming class is
        # constructed without an explicit model. Passing one to a class
        # overrides the default for that instance only.
        "scope": "global defaults, overridable per class",
    }

    try:
        llm, _ = initialize_model()
        result["llm"] = {
            "provider": _PROVIDER_BY_CLASS.get(
                type(llm).__name__, type(llm).__name__
            ),
            "model": llm.get_model_name(),
        }
    except Exception as e:
        result["llm"] = {"error": str(e)}
    result["llm"][
        "used_by"
    ] = "metrics, synthesizer, simulator, benchmarks, prompt optimizer"

    try:
        embedder = initialize_embedding_model()
        result["embeddings"] = {
            "provider": _PROVIDER_BY_CLASS.get(
                type(embedder).__name__, type(embedder).__name__
            ),
            "model": embedder.get_model_name(),
        }
    except Exception as e:
        result["embeddings"] = {"error": str(e)}
    result["embeddings"]["used_by"] = (
        "synthesizer context construction only "
        "(override with embedder=... in ContextConstructionConfig)"
    )

    return result


def _setting_sources_section() -> Dict[str, Any]:
    """The places a setting can come from, highest precedence first.

    Mirrors the runtime merge: process env wins per variable, then dotenv
    files (in `dotenv_search_paths()` order), then the legacy JSON keystore,
    then pydantic defaults.
    """
    settings = get_settings()
    dotenv_disabled = parse_bool(
        os.getenv("DEEPEVAL_DISABLE_DOTENV"), default=False
    )
    legacy_disabled = parse_bool(
        os.getenv("DEEPEVAL_DISABLE_LEGACY_KEYFILE"), default=False
    )
    keystore_path, keystore_data = _legacy_keystore()

    sources: List[Dict[str, Any]] = [
        {"source": "process environment", "status": "always active"}
    ]

    for path in _dotenv_paths():  # highest precedence first
        if dotenv_disabled:
            status = "skipped (DEEPEVAL_DISABLE_DOTENV is set)"
        else:
            status = "found" if path.is_file() else "not found"
        sources.append({"source": path.name, "status": status})

    if legacy_disabled:
        keystore_status = "skipped (DEEPEVAL_DISABLE_LEGACY_KEYFILE is set)"
    elif keystore_path.is_file():
        keystore_status = f"found ({len(keystore_data)} key(s))"
    else:
        keystore_status = "not found"
    sources.append(
        {
            "source": f"{keystore_path} (JSON keystore)",
            "status": keystore_status,
            "keys": sorted(keystore_data.keys()),
        }
    )

    sources.append({"source": "built-in defaults", "status": "fallback"})

    for rank, source in enumerate(sources, start=1):
        source["rank"] = rank

    return {
        "note": "resolved independently per variable; lowest rank wins",
        "env_dir": str(resolve_env_dir()),
        "precedence": sources,
        "default_save": settings.DEEPEVAL_DEFAULT_SAVE,
    }


def _configured_settings_section() -> List[Dict[str, Any]]:
    settings = get_settings()
    dotenv_values = [
        (p, read_dotenv_file(p)) for p in _dotenv_paths() if p.is_file()
    ]
    _, legacy_data = _legacy_keystore()
    json_key_map = _env_key_to_json_key()

    rows: List[Dict[str, Any]] = []
    for name in sorted(type(settings).model_fields):
        if not any(marker in name for marker in _RELEVANT_MARKERS):
            continue
        value = getattr(settings, name, None)
        if value is None:
            continue
        source = _resolve_source(name, dotenv_values, legacy_data, json_key_map)
        if source is None:
            continue  # built-in default; not user-configured
        rows.append(
            {"name": name, "value": _display_value(value), "source": source}
        )
    return rows


def diagnose_command(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output the report as JSON (machine-readable, secrets masked).",
    ),
) -> None:
    """Show the effective DeepEval configuration and where it came from."""
    report = {
        "deepeval_version": _deepeval_version(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "default_models": _models_section(),
        "configured_settings": _configured_settings_section(),
        "setting_sources": _setting_sources_section(),
        "confident_ai": _confident_section(),
    }

    if json_output:
        typer.echo(json_lib.dumps(report, indent=2))
        return

    console = Console()

    def _kv_table(title: str) -> Table:
        table = Table(
            title=title,
            title_justify="left",
            title_style="bold rgb(106,0,255)",
            show_header=False,
            box=box.ROUNDED,
            border_style="dim",
            padding=(0, 1),
            min_width=60,
        )
        table.add_column("Field", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        return table

    console.print(
        f"\n[bold]DeepEval[/bold] [dim]{report['deepeval_version']} · "
        f"Python {report['python_version']}[/dim]\n"
    )

    # Default models (global, overridable per class)
    models = report["default_models"]
    table = _kv_table("Default models")
    for label, key in (("LLM", "llm"), ("Embeddings", "embeddings")):
        info = models[key]
        if "error" in info:
            value = f"[red]⚠ unusable until fixed: {info['error']}[/red]"
        else:
            value = f"[bold]{info['provider']}[/bold] — {info['model']}"
        table.add_row(label, f"{value}\n[dim]used by {info['used_by']}[/dim]")
    console.print(table)
    console.print(
        "[dim]Global defaults: apply whenever a class is constructed without "
        "an explicit model.[/dim]\n"
    )

    # Configured settings and their winning sources
    rows = report["configured_settings"]
    if rows:
        table = Table(
            title="Configured settings",
            title_justify="left",
            title_style="bold rgb(106,0,255)",
            box=box.ROUNDED,
            border_style="dim",
            header_style="bold",
            padding=(0, 1),
            min_width=60,
        )
        table.add_column("Setting", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_column("Winning source", overflow="fold", style="dim")
        for row in rows:
            table.add_row(row["name"], row["value"], row["source"])
        console.print(table)
    else:
        console.print(
            "[dim]No model/Confident settings configured; "
            "everything is on built-in defaults.[/dim]"
        )

    # Setting sources: subordinate legend for the table above, indented.
    env = report["setting_sources"]
    table = Table(
        title="Where settings can come from (highest precedence first)",
        title_justify="left",
        title_style="dim bold",
        box=box.ROUNDED,
        border_style="dim",
        header_style="bold",
        padding=(0, 1),
        min_width=56,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Source", style="bold", no_wrap=True)
    table.add_column("Status", overflow="fold")
    for source in env["precedence"]:
        status = source["status"]
        if status.startswith("found"):
            status_markup = f"[green]{status}[/green]"
        elif status.startswith("skipped"):
            status_markup = f"[yellow]{status}[/yellow]"
        elif status == "not found":
            status_markup = "[dim]not found[/dim]"
        else:
            status_markup = f"[dim]{status}[/dim]"
        table.add_row(str(source["rank"]), source["source"], status_markup)
    console.print(Padding(table, (0, 0, 0, 3)))
    console.print(
        Padding(
            "[dim]Each variable is resolved independently: the "
            'highest-ranked source that defines it wins (the "winning '
            f'source" above). Dotenv files are looked up in '
            f"{env['env_dir']}. Login and set-* commands persist to "
            f"{env['default_save'] or '.env.local (default)'}.[/dim]\n",
            (0, 0, 0, 3),
        )
    )

    # Confident AI
    confident = report["confident_ai"]
    table = _kv_table("Confident AI")
    table.add_row(
        "Status",
        (
            "[green]● Logged in[/green]"
            if confident["logged_in"]
            else "[yellow]○ Not logged in[/yellow] [dim](run `deepeval login`)[/dim]"
        ),
    )
    if confident["api_key"]:
        table.add_row("API key", confident["api_key"])
    if confident["region"]:
        table.add_row(
            "Data region",
            f"{confident['region']} [dim]({confident['region_source']})[/dim]",
        )
    else:
        table.add_row(
            "Data region", f"[dim]({confident['region_source']})[/dim]"
        )
    table.add_row("API endpoint", confident["api_base_url"])
    if confident["otel_url"]:
        otel_flagged = any(
            "OTEL" in warning for warning in confident["warnings"]
        )
        table.add_row(
            "OTEL endpoint",
            (
                f"[yellow]{confident['otel_url']} ⚠[/yellow]"
                if otel_flagged
                else confident["otel_url"]
            ),
        )
    console.print(table)
    for warning in confident["warnings"]:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
