"""Probe registered OpenAI Chat Completions model capabilities.

Usage:
    OPENAI_API_KEY=... python scripts/check_openai_model_capabilities.py
    OPENAI_API_KEY=... python scripts/check_openai_model_capabilities.py gpt-5.4 gpt-5.5
    OPENAI_API_KEY=... python scripts/check_openai_model_capabilities.py --all-registry-models

By default this checks the current frontier models whose registry flags have
changed recently. Pass explicit model names or --all-registry-models to expand
the probe.
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Callable

from deepeval.models.llms.constants import OPENAI_MODELS_DATA


DEFAULT_MODELS = ("gpt-5.4", "gpt-5.5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI model support for logprobs and JSON mode."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help=(
            "OpenAI model names to probe. Defaults to "
            f"{', '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--all-registry-models",
        action="store_true",
        help="Probe every model listed in deepeval's OPENAI_MODELS_DATA.",
    )
    return parser.parse_args()


def select_models(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all_registry_models:
        return tuple(OPENAI_MODELS_DATA.keys())
    if args.models:
        return tuple(args.models)
    return DEFAULT_MODELS


def registry_expectations(model: str) -> dict[str, Any]:
    model_data = OPENAI_MODELS_DATA.get(model)
    return {
        "registered": model in OPENAI_MODELS_DATA,
        "supports_log_probs": model_data.supports_log_probs,
        "supports_json": model_data.supports_json,
        "supports_structured_outputs": model_data.supports_structured_outputs,
        "supports_temperature": model_data.supports_temperature,
    }


def summarize_response(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "content": getattr(message, "content", None),
        "has_logprobs": getattr(choice, "logprobs", None) is not None,
        "usage": (
            response.usage.model_dump()
            if hasattr(response.usage, "model_dump")
            else response.usage
        ),
    }


def run_check(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        response = call()
        return {
            "parameter_accepted": True,
            "succeeded": True,
            "response": summarize_response(response),
        }
    except Exception as exc:
        return {
            "parameter_accepted": False,
            "succeeded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_json_mode_check(call: Callable[[], Any]) -> dict[str, Any]:
    summary: dict[str, Any] | None = None
    try:
        response = call()
        summary = summarize_response(response)
        content = summary["content"] or ""
        parsed_json = json.loads(content)
        return {
            "parameter_accepted": True,
            "succeeded": True,
            "response": summary,
            "parsed_json": parsed_json,
        }
    except json.JSONDecodeError as exc:
        return {
            "parameter_accepted": True,
            "succeeded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "response": summary,
        }
    except Exception as exc:
        return {
            "parameter_accepted": False,
            "succeeded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def probe_model(client: Any, model: str) -> dict[str, Any]:
    return {
        "registry": registry_expectations(model),
        "logprobs": run_check(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with exactly one short sentence.",
                    }
                ],
                max_completion_tokens=32,
                logprobs=True,
                top_logprobs=1,
            ),
        ),
        "json_mode": run_json_mode_check(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Return only valid JSON. Do not include markdown. "
                            "Use this exact schema: "
                            '{"model": string, '
                            '"supports_json_mode": boolean}.'
                        ),
                    }
                ],
                max_completion_tokens=256,
                response_format={"type": "json_object"},
            ),
        ),
    }


def main() -> None:
    args = parse_args()
    openai = importlib.import_module("openai")
    client = openai.OpenAI()
    results = {
        model: probe_model(client, model) for model in select_models(args)
    }
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
