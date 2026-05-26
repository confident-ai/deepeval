"""Compile metric template .txt files into metric_templates/templates.json.

Every template lives at ``**/templates/<ClassName>/<method>.txt`` under ``deepeval/``.
Fragments live at ``metric_templates/fragments/<name>.txt``.

Usage:
    python scripts/compile_metric_templates.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "deepeval"
TEMPLATES_JSON = PACKAGE_ROOT / "metric_templates" / "templates.json"
FRAGMENTS_DIR = PACKAGE_ROOT / "metric_templates" / "fragments"


def _collect_from_disk() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    classes: dict[str, dict[str, str]] = defaultdict(dict)
    for path in PACKAGE_ROOT.rglob("templates/*/*.txt"):
        if path.parent.parent.name != "templates":
            continue
        class_name = path.parent.name
        classes[class_name][path.stem] = path.read_text(encoding="utf-8")

    fragments = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(FRAGMENTS_DIR.glob("*.txt"))
    }
    return dict(classes), fragments


def main() -> None:
    classes, fragments = _collect_from_disk()

    existing: dict = {}
    if TEMPLATES_JSON.is_file():
        existing = json.loads(TEMPLATES_JSON.read_text(encoding="utf-8"))
    existing_keys = list(existing.keys())

    ordered_keys: list[str] = []
    for key in existing_keys:
        if key == "_fragments":
            if fragments:
                ordered_keys.append("_fragments")
        elif key in classes:
            ordered_keys.append(key)

    for key in sorted(classes):
        if key not in ordered_keys:
            ordered_keys.append(key)
    if fragments and "_fragments" not in ordered_keys:
        ordered_keys.append("_fragments")

    bundle: dict = {}
    for key in ordered_keys:
        if key == "_fragments":
            bundle["_fragments"] = fragments
        else:
            methods = classes[key]
            if isinstance(existing.get(key), dict):
                method_order = [m for m in existing[key] if m in methods]
                method_order += sorted(m for m in methods if m not in method_order)
            else:
                method_order = sorted(methods)
            bundle[key] = {m: methods[m] for m in method_order}

    TEMPLATES_JSON.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {TEMPLATES_JSON}")


if __name__ == "__main__":
    main()
