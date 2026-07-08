#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field, field_validator

try:
    from generate import (
        gh_request,
        get_ai_model,
        sanitize_for_multimodal_sentinel,
        clean_title,
        classify,
        OWNER,
        REPO,
        CATEGORY_ORDER,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from generate import (
        gh_request,
        get_ai_model,
        sanitize_for_multimodal_sentinel,
        clean_title,
        classify,
        OWNER,
        REPO,
        CATEGORY_ORDER,
    )

HEADER = "New to `deepeval`? Get started [here](https://deepeval.com/docs/introduction)."
FOOTER = "A huge thank you to everyone who contributed to this release ❤️"

CATEGORY_EMOJI = {
    "Backward Incompatible Change": "💥",
    "New Feature": "✨",
    "Experimental Feature": "🧪",
    "Improvement": "⚡",
    "Bug Fix": "🐛",
    "Security": "🔒",
}

TS_PREFIXES = ("typescript/",)
PY_PREFIXES = ("deepeval/",)
PY_FILES = ("pyproject.toml", "poetry.lock")

BATCH_MAX = 60


@dataclass
class PrLine:
    number: int
    title: str
    login: str
    url: str


@dataclass
class ParsedNotes:
    prs: List[PrLine] = field(default_factory=list)
    new_contributors: Dict[int, str] = field(default_factory=dict)


class CleanedItem(BaseModel):
    pr_number: int
    entry: str = Field(..., min_length=1, max_length=500)
    category: Literal[
        "Backward Incompatible Change",
        "New Feature",
        "Experimental Feature",
        "Improvement",
        "Bug Fix",
        "Security",
    ]

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value):
        return value if value in CATEGORY_ORDER else "Improvement"


class CleanedNotes(BaseModel):
    items: List[CleanedItem]


WHATS_CHANGED_LINE_RE = re.compile(
    r"^\*\s+(?P<title>.+?)\s+by\s+@(?P<login>[A-Za-z0-9-]+)\s+in\s+"
    r"(?P<url>https?://\S+/pull/(?P<num>\d+))\s*$"
)
NEW_CONTRIB_RE = re.compile(
    r"^\*\s+@(?P<login>[A-Za-z0-9-]+)\s+made their first contribution in\s+"
    r"\S+/pull/(?P<num>\d+)"
)

SECTION_WHATS_CHANGED = "## What's Changed"
SECTION_NEW_CONTRIB = "## New Contributors"


def parse_release_body(body: str) -> ParsedNotes:
    parsed = ParsedNotes()
    section = None
    for line in body.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            section = stripped
            continue

        if section == SECTION_WHATS_CHANGED:
            match = WHATS_CHANGED_LINE_RE.match(stripped)
            if match:
                parsed.prs.append(
                    PrLine(
                        number=int(match.group("num")),
                        title=match.group("title").strip(),
                        login=match.group("login"),
                        url=match.group("url"),
                    )
                )
        elif section == SECTION_NEW_CONTRIB:
            match = NEW_CONTRIB_RE.match(stripped)
            if match:
                parsed.new_contributors[int(match.group("num"))] = match.group("login")

    return parsed


def _is_ts(path: str) -> bool:
    return any(path.startswith(p) for p in TS_PREFIXES)


def _is_py(path: str) -> bool:
    return any(path.startswith(p) for p in PY_PREFIXES) or path in PY_FILES


def classify_sdk(files: List[str]) -> str:
    ts = any(_is_ts(f) for f in files)
    py = any(_is_py(f) for f in files)
    if ts and not py:
        return "typescript"
    if py and not ts:
        return "python"
    return "shared"


def pr_scope(number: int) -> str:
    ts = py = False
    page = 1
    try:
        while True:
            batch = gh_request(
                f"/repos/{OWNER}/{REPO}/pulls/{number}/files?per_page=100&page={page}"
            )
            if not isinstance(batch, list) or not batch:
                break
            for item in batch:
                name = item.get("filename") or ""
                ts = ts or _is_ts(name)
                py = py or _is_py(name)
            if (ts and py) or len(batch) < 100:
                break
            page += 1
    except urllib.error.HTTPError as exc:
        log(f"warning: could not fetch files for #{number} ({exc}); treating as shared")
        return "shared"

    if ts and not py:
        return "typescript"
    if py and not ts:
        return "python"
    return "shared"


def select_in_scope(prs: List[PrLine], sdk: str) -> List[PrLine]:
    kept = []
    for pr in prs:
        scope = pr_scope(pr.number)
        if scope in (sdk, "shared"):
            kept.append(pr)
        log(f"#{pr.number}: {scope} -> {'keep' if scope in (sdk, 'shared') else 'drop'}")
    return kept


def build_cleanup_prompt(prs: List[PrLine]) -> str:
    listing = "\n".join(f"{pr.number}: {pr.title}" for pr in prs)
    return f"""
You are writing release notes for an open-source developer tool.

You are given a list of merged pull requests as "<pr_number>: <raw title>".
For EACH pr_number, produce one item with:
- pr_number: the same number, unchanged.
- entry: one clean, user-facing release-note line. Plain text, an action verb
  ("Add", "Fix", "Improve", "Support"...), 1 sentence, no markdown headers/lists,
  no PR numbers, no URLs, no author names. Inline `backticks` for code are fine.
- category: the single best match from the allowed categories.

Return exactly one item per input pr_number. Do not invent pr_numbers.

Allowed categories:
- Backward Incompatible Change
- New Feature
- Experimental Feature
- Improvement
- Bug Fix
- Security

Pull requests:
{listing}
""".strip()


def ai_cleanup(model, prs: List[PrLine]) -> Dict[int, Tuple[str, str]]:
    cleaned: Dict[int, Tuple[str, str]] = {}
    total_cost = 0.0
    for start in range(0, len(prs), BATCH_MAX):
        chunk = prs[start : start + BATCH_MAX]
        prompt = sanitize_for_multimodal_sentinel(build_cleanup_prompt(chunk))
        parsed, cost = model.generate(prompt, schema=CleanedNotes)
        assert isinstance(parsed, CleanedNotes)
        total_cost += cost or 0.0
        for item in parsed.items:
            cleaned[item.pr_number] = (item.entry.strip(), item.category)
    log(f"AI cleanup cost: ${total_cost:.4f}")
    return cleaned


def fallback_entry(pr: PrLine) -> Tuple[str, str]:
    return clean_title(pr.title), classify(pr.title, "")


def render_markdown(
    prs: List[PrLine],
    cleaned: Dict[int, Tuple[str, str]],
    new_contributors: Dict[int, str],
) -> str:
    buckets: Dict[str, List[str]] = {c: [] for c in CATEGORY_ORDER}
    for pr in prs:
        entry, category = cleaned.get(pr.number) or fallback_entry(pr)
        if category not in buckets:
            category = "Improvement"
        buckets[category].append(f"* {entry} (#{pr.number} by @{pr.login})")

    parts: List[str] = [HEADER, ""]

    if any(buckets[c] for c in CATEGORY_ORDER):
        for category in CATEGORY_ORDER:
            lines = buckets[category]
            if not lines:
                continue
            emoji = CATEGORY_EMOJI.get(category, "")
            parts.append(f"### {emoji} {category}".strip())
            parts.extend(lines)
            parts.append("")
    else:
        parts.append("_No user-facing changes in this release._")
        parts.append("")

    contributors = [f"@{login}" for login in new_contributors.values()]
    if contributors:
        parts.append("**New contributors:** " + ", ".join(contributors))
        parts.append("")

    parts.append(FOOTER)
    return "\n".join(parts).rstrip() + "\n"


_SILENT = False


def log(message: str) -> None:
    if not _SILENT:
        print(message, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Clean up and SDK-scope GitHub release notes from stdin."
    )
    ap.add_argument("--sdk", required=True, choices=["python", "typescript"])
    ap.add_argument("--ai-model", default="gpt-5.2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--silent", action="store_true")
    return ap.parse_args()


def main() -> int:
    global _SILENT
    args = parse_args()
    _SILENT = args.silent

    parsed = parse_release_body(sys.stdin.read())
    log(f"Parsed {len(parsed.prs)} PRs from release body.")

    in_scope = select_in_scope(parsed.prs, args.sdk)
    log(f"{len(in_scope)} PRs in scope for '{args.sdk}'.")

    in_scope_numbers = {pr.number for pr in in_scope}
    new_contributors = {
        num: login
        for num, login in parsed.new_contributors.items()
        if num in in_scope_numbers
    }

    cleaned: Dict[int, Tuple[str, str]] = {}
    if in_scope and not (args.dry_run or args.no_ai):
        try:
            cleaned = ai_cleanup(get_ai_model(args.ai_model), in_scope)
        except Exception as exc:
            log(f"warning: AI cleanup failed ({exc}); using deterministic titles.")
            cleaned = {}

    sys.stdout.write(render_markdown(in_scope, cleaned, new_contributors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
