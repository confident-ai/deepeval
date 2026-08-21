"""Compile template .txt files into per-feature templates.json bundles.

Templates live under a feature's source tree in one of two layouts:

* Flat (default): the directory holds a ``class.txt`` marker naming the owning
  class, and the sibling ``<method>.txt`` files are that class's methods —
  ``**/templates/class.txt`` + ``**/templates/<method>.txt``.
* Nested (multi-class, e.g. ``dag``): the directory has no ``class.txt`` marker
  and instead groups methods under one subfolder per class —
  ``**/templates/<ClassName>/<method>.txt``.

Optional shared snippets for a feature live at
``templates/<feature>/fragments/<name>.txt`` (metrics only today).

Each feature's compiled bundle is written to BOTH the Python package and the
TypeScript package so the two stay in sync.

Usage:
    python scripts/compile_metric_templates.py            # all features
    python scripts/compile_metric_templates.py metrics    # one feature
    python scripts/compile_metric_templates.py simulator
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "deepeval"
TS_TEMPLATES_ROOT = REPO_ROOT / "typescript" / "src" / "templates"


class FeatureConfig(NamedTuple):
    name: str
    # Root to scan for **/templates/ sources (must not include other features).
    sources_root: Path
    # Optional shared-snippet dir; None when the feature has no snippets.
    fragments_dir: Optional[Path] = None


FEATURES: dict[str, FeatureConfig] = {
    "metrics": FeatureConfig(
        name="metrics",
        sources_root=PACKAGE_ROOT / "metrics",
        fragments_dir=PACKAGE_ROOT / "templates" / "metrics" / "fragments",
    ),
    "simulator": FeatureConfig(
        name="simulator",
        sources_root=PACKAGE_ROOT / "simulator",
        fragments_dir=None,
    ),
}


def _py_json(feature: str) -> Path:
    return PACKAGE_ROOT / "templates" / feature / "templates.json"


def _ts_json(feature: str) -> Path:
    return TS_TEMPLATES_ROOT / feature / "templates.json"


def _collect_from_disk(
    config: FeatureConfig,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    classes: dict[str, dict[str, str]] = defaultdict(dict)
    bundle_dir = _py_json(config.name).parent

    for templates_dir in config.sources_root.rglob("templates"):
        if not templates_dir.is_dir():
            continue
        # Don't descend into the compiled-bundle dir.
        if templates_dir == bundle_dir:
            continue

        marker = templates_dir / "class.txt"
        if marker.is_file():
            class_name = marker.read_text(encoding="utf-8").strip()
            for path in templates_dir.glob("*.txt"):
                if path.name == "class.txt":
                    continue
                classes[class_name][path.stem] = path.read_text(
                    encoding="utf-8"
                )
        else:
            for sub in templates_dir.iterdir():
                if not sub.is_dir():
                    continue
                for path in sub.glob("*.txt"):
                    classes[sub.name][path.stem] = path.read_text(
                        encoding="utf-8"
                    )

    fragments: dict[str, str] = {}
    if config.fragments_dir is not None and config.fragments_dir.is_dir():
        fragments = {
            path.stem: path.read_text(encoding="utf-8")
            for path in sorted(config.fragments_dir.glob("*.txt"))
        }
    return dict(classes), fragments


def build_bundle(feature: str = "metrics") -> dict:
    """Build the templates bundle from the .txt sources on disk.

    Preserves the key/method ordering of the existing ``templates.json`` so a
    no-op recompile produces a byte-identical file.
    """
    if feature not in FEATURES:
        raise ValueError(
            f"Unknown feature {feature!r}; expected one of "
            f"{', '.join(sorted(FEATURES))}."
        )
    config = FEATURES[feature]
    classes, fragments = _collect_from_disk(config)
    templates_json = _py_json(feature)

    existing: dict = {}
    if templates_json.is_file():
        existing = json.loads(templates_json.read_text(encoding="utf-8"))
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


def compile_feature(feature: str) -> None:
    content = render_bundle_json(build_bundle(feature))
    for path in (_py_json(feature), _ts_json(feature)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Updated {path}")


def main(argv: Optional[list[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    features = args if args else list(FEATURES)
    for feature in features:
        compile_feature(feature)


if __name__ == "__main__":
    main()
