"""Compile metric template .txt files into templates/metrics/templates.json.

Templates live under ``**/templates/`` directories in one of two layouts:

* Flat (default): the directory holds a ``class.txt`` marker naming the owning
  class, and the sibling ``<method>.txt`` files are that class's methods —
  ``**/templates/class.txt`` + ``**/templates/<method>.txt``.
* Nested (multi-class, e.g. ``dag``): the directory has no ``class.txt`` marker
  and instead groups methods under one subfolder per class —
  ``**/templates/<ClassName>/<method>.txt``.

Fragments live at ``templates/metrics/fragments/<name>.txt``.

The compiled bundle is written to BOTH the Python package and the TypeScript
package so the two stay in sync.

Usage:
    python scripts/compile_metric_templates.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "deepeval"
FEATURE = "metrics"
# Compiled bundle is emitted to both packages.
TEMPLATES_JSON = PACKAGE_ROOT / "templates" / FEATURE / "templates.json"
TS_TEMPLATES_JSON = (
    REPO_ROOT / "typescript" / "src" / "templates" / FEATURE / "templates.json"
)
FRAGMENTS_DIR = PACKAGE_ROOT / "templates" / FEATURE / "fragments"


def _collect_from_disk() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    classes: dict[str, dict[str, str]] = defaultdict(dict)
    for templates_dir in PACKAGE_ROOT.rglob("templates"):
        if not templates_dir.is_dir():
            continue
        # Don't descend into the compiled-bundle dir (it holds templates.json
        # and the fragments folder, not raw .txt sources).
        if templates_dir == TEMPLATES_JSON.parent:
            continue

        marker = templates_dir / "class.txt"
        if marker.is_file():
            # Flat layout: class name comes from the marker; siblings are methods.
            class_name = marker.read_text(encoding="utf-8").strip()
            for path in templates_dir.glob("*.txt"):
                if path.name == "class.txt":
                    continue
                classes[class_name][path.stem] = path.read_text(
                    encoding="utf-8"
                )
        else:
            # Nested layout: one subfolder per class (multi-class metrics).
            for sub in templates_dir.iterdir():
                if not sub.is_dir():
                    continue
                for path in sub.glob("*.txt"):
                    classes[sub.name][path.stem] = path.read_text(
                        encoding="utf-8"
                    )

    fragments = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(FRAGMENTS_DIR.glob("*.txt"))
    }
    return dict(classes), fragments


def build_bundle() -> dict:
    """Build the templates bundle from the .txt sources on disk.

    Preserves the key/method ordering of the existing ``templates.json`` so a
    no-op recompile produces a byte-identical file.
    """
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
                method_order += sorted(
                    m for m in methods if m not in method_order
                )
            else:
                method_order = sorted(methods)
            bundle[key] = {m: methods[m] for m in method_order}

    return bundle


def render_bundle_json(bundle: dict) -> str:
    """Serialize the bundle exactly as it is written to ``templates.json``."""
    return json.dumps(bundle, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    content = render_bundle_json(build_bundle())
    for path in (TEMPLATES_JSON, TS_TEMPLATES_JSON):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
