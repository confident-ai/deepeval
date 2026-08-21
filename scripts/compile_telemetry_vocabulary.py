"""Compile Python's telemetry vocabulary into a committed JSON artifact.

Both SDKs report into one PostHog project, where a key that differs between them
forks a series silently rather than failing. Python is the older reporter and
therefore the reference: every wire string it can send is written to
``typescript/test/test-core/telemetry-vocabulary.json``, which a Jest test
asserts the TypeScript enums against. Committed so that suite needs no Python.

Usage:
    python scripts/compile_telemetry_vocabulary.py
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCABULARY_JSON = (
    REPO_ROOT
    / "typescript"
    / "test"
    / "test-core"
    / "telemetry-vocabulary.json"
)


def _values(enum_class: Any) -> list:
    return sorted(member.value for member in enum_class)


def build() -> Dict[str, Any]:
    from deepeval.telemetry import events, identity, properties
    from deepeval.tracing.integrations import Integration

    return {
        "_meta": {
            "doNotEdit": True,
            "generated_by": "scripts/compile_telemetry_vocabulary.py",
            "source": "deepeval/telemetry/",
        },
        "schemaVersion": events.TELEMETRY_SCHEMA_VERSION,
        "runIdEnvVar": events.TELEMETRY_RUN_ID_ENV_VAR,
        "events": _values(events.Event),
        "entrypoints": _values(events.Entrypoint),
        "features": _values(events.Feature),
        "integrations": _values(Integration),
        "props": _values(properties.Prop),
        # Field name -> wire key. The TypeScript payload type uses camelCase
        # fields, so only the values are compared; the keys document the mapping.
        "propsByField": {
            field: prop.value
            for field, prop in properties._FIELD_TO_PROP.items()
        },
        "payloadFields": [
            field.name for field in fields(properties.EventProperties)
        ],
        "languages": _values(properties.Language),
        "runtimes": _values(properties.Runtime),
        "userStatuses": _values(properties.UserStatus),
        "outcomes": _values(properties.Outcome),
        "turnKinds": _values(properties.TurnKind),
        "flushReasons": _values(properties.FlushReason),
        "loginPromptSurfaces": _values(properties.LoginPromptSurface),
        "loginOutcomes": _values(properties.LoginOutcome),
        "loginMethods": _values(properties.LoginMethod),
        "telemetryKeys": _values(properties.TelemetryKey),
        "sentinels": {
            "customProvider": properties.CUSTOM_PROVIDER,
            "unknownModel": properties.UNKNOWN_MODEL,
            "unknownCliCommand": properties.UNKNOWN_CLI_COMMAND,
        },
        "identityFileName": identity.TELEMETRY_DATA_FILE,
    }


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    VOCABULARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    VOCABULARY_JSON.write_text(render(), encoding="utf-8")
    print(f"Wrote {VOCABULARY_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
