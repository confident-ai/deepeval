#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from rich import print
from rich.console import Console, Group
from rich.markup import escape
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.live import Live

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


#################
# Configuration #
#################

OWNER = "confident-ai"
REPO = "deepeval"

START_MARKER = "<!-- DeepEval release notes start -->"

CATEGORY_ORDER = [
    "Backward Incompatible Change",
    "New Feature",
    "Experimental Feature",
    "Improvement",
    "Bug Fix",
    "Security",
]

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_INDEX = {name: i for i, name in enumerate(MONTH_NAMES, start=1)}
AI_MAX_DIFF_LENGTH = 12000  # max chars for diff
CLEAR_PROGRESS_BAR_ON_COMPLETION = False


##############
# Data types #
##############


@dataclass
class Commit:
    sha: str
    subject: str


@dataclass
class Pull:
    number: int
    title: str
    body: str
    merged_at: str
    html_url: str
    user_login: str
    user_html_url: str
    diff_url: str


class AiReleaseNote(BaseModel):
    entry: str = Field(
        ...,
        description="User-facing changelog entry. Plain text. No markdown. No PR numbers/links.",
        min_length=10,
        max_length=500,
    )
    category: str
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional internal notes; not written to changelog.",
        max_length=400,
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, category: str) -> str:
        if category not in CATEGORY_ORDER:
            raise ValueError(f"category must be one of: {CATEGORY_ORDER}")
        return category


class AiMonthSummary(BaseModel):
    summary: str = Field(
        ...,
        description="Short prose summary for the month. Plain text. No lists. No headings.",
        min_length=40,
        max_length=700,
    )


#######################
# Git and PR parsing  #
#######################

PR_NUM_RE = re.compile(r"\(#(\d+)\)|pull request #(\d+)", re.IGNORECASE)
MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #(\d+)\b", re.IGNORECASE)
user_cache: Dict[str, Tuple[str, str]] = (
    {}
)  # maps login to (display_name, html_url)
tag_to_date: Dict[str, str] = {}

###################################
# Changelog index and MDX parsing #
###################################

ChangelogIndex = Dict[str, Dict[str, Dict[str, Dict[int, str]]]]
# month -> category -> version -> pr_number -> bullet_line

MONTH_RE = re.compile(r"^##\s+(.+?)\s*$")
CATEGORY_RE = re.compile(r"^###\s+(.+?)\s*$")
VERSION_RE = re.compile(r"^####\s+(v[0-9].+?)\s*$")

# Bullet PR extraction:
# - Prefer the stable marker (lets humans edit the visible link/text)
# - Fall back to parsing the link if the marker is missing
BULLET_PR_RE = re.compile(r"\[#(\d+)\]\(")
BULLET_PR_MARKER_RE = re.compile(r"<!--\s*pr:(\d+)\s*-->")
BULLET_TAIL_RE = re.compile(
    r"\s*\(\[#\d+\]\([^)]+\)\)\s*<!--\s*pr:\d+\s*-->.*$"
)

# Optional ignore list to be placed right after START_MARKER to avoid confusing the parser:
# add a list of PR numbers you would like to be excluded from the generated changelog.
# <!-- changelog-ignore:
# - 1234
# - 5678
# -->
IGNORE_BLOCK_TOP_RE = re.compile(
    r"(?is)^\s*<!--\s*changelog-ignore:.*?-->\s*\n*"
)
IGNORE_BLOCK_ANY_RE = re.compile(r"(?is)<!--\s*changelog-ignore:(.*?)-->")

###############
# Git helpers #
###############


def sh(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def git_tag_date_ymd(tag: str) -> str:
    if tag not in tag_to_date:
        date_value = sh(["git", "log", "-1", "--format=%cs", tag])
        tag_to_date[tag] = date_value
    return tag_to_date[tag]


def get_prev_tag(tag: str) -> str:
    return sh(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*", f"{tag}^"]
    )


def list_tags_between(from_tag: str, to_tag: str) -> List[str]:
    # Ordered by tag date ascending
    # Uses creatordate which works for lightweight tags too.
    raw = sh(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)%09%(creatordate:short)",
            "--sort=creatordate",
            "refs/tags/v*",
        ]
    )
    tags: List[Tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        tag, date = line.split("\t", 1)
        tags.append((tag.strip(), date.strip()))
    # filter in [from..to] by order in this sorted list
    tag_names = [tag for tag, _ in tags]
    if from_tag not in tag_names or to_tag not in tag_names:
        raise SystemExit(
            f"from/to tag not found in local tags: {from_tag} -> {to_tag}"
        )
    from_index = tag_names.index(from_tag)
    to_index = tag_names.index(to_tag)
    if from_index > to_index:
        from_index, to_index = to_index, from_index
    return tag_names[from_index : to_index + 1]


def list_all_tags() -> List[str]:
    """Return all version tags sorted by tag creation date (ascending)."""
    raw = sh(
        [
            "git",
            "for-each-ref",
            "--sort=creatordate",
            "--format",
            "%(refname:short)\t%(creatordate:short)",
            "refs/tags/v*",
        ]
    )
    tags: List[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        tag = line.split("\t", 1)[0].strip()
        if tag:
            tags.append(tag)
    return tags


def list_tags_for_year(year: int) -> List[str]:
    """
    Return all tags whose effective tag date falls within `year`.
    """
    all_tags = list_all_tags()
    out: List[str] = []
    for tag in all_tags:
        ymd = git_tag_date_ymd(tag)
        if ymd.startswith(f"{year}-"):
            out.append(tag)
    # keep chronological order (oldest -> newest)
    out.sort(key=lambda t: git_tag_date_ymd(t))
    return out


def latest_tag() -> str:
    return sh(["git", "describe", "--tags", "--abbrev=0", "--match", "v*"])


def commits_in_range(base: str, head: str) -> List[Commit]:
    # get sha and subject for commit subjects in range
    raw = sh(
        [
            "git",
            "log",
            "--first-parent",
            "--merges",
            "--format=%H%x00%s",
            f"{base}..{head}",
        ]
    )
    commits: List[Commit] = []
    for line in raw.splitlines():
        if "\x00" not in line:
            continue
        sha, subj = line.split("\x00", 1)
        commits.append(Commit(sha=sha.strip(), subject=subj.strip()))
    return commits


def extract_pr_numbers(commits: Iterable[Commit]) -> Dict[int, Commit]:
    # Map PR to representative commit
    prs: Dict[int, Commit] = {}
    for commit in commits:
        subj_match = PR_NUM_RE.search(commit.subject)
        if not subj_match:
            continue
        n = subj_match.group(1) or subj_match.group(2)
        if not n:
            continue
        pr_num = int(n)
        # prefer merge commit subjects if multiple commits mention same PR
        if pr_num not in prs:
            prs[pr_num] = commit
        else:
            if MERGE_SUBJECT_RE.match(
                commit.subject
            ) and not MERGE_SUBJECT_RE.match(prs[pr_num].subject):
                prs[pr_num] = commit
    return prs


def offline_pr_title_from_merge_commit(
    commit_sha: str, fallback_subject: str
) -> str:
    """
    GitHub merge commits look like:
      Merge pull request #1234 from ...

      PR Title here

    So, we don't need to use the api to get the PR title from the commit message body
    """
    if not MERGE_SUBJECT_RE.match(fallback_subject):
        return fallback_subject

    full = sh(["git", "show", "-s", "--format=%B", commit_sha])
    lines = [ln.rstrip() for ln in full.splitlines()]
    # the first line is merge subject, so find first non empty line after it
    for ln in lines[1:]:
        if ln.strip():
            return ln.strip()
    return fallback_subject


def stitch_truncated_title(title: str, body: str) -> str:
    t = (title or "").strip()
    if not body:
        return t

    # If title ends with ellipsis, try to append the first non-empty line of the body.
    if t.endswith("…") or t.endswith("..."):
        first_line = next(
            (ln.strip() for ln in body.splitlines() if ln.strip()), ""
        )
        if first_line:
            t2 = t[:-1].rstrip() if t.endswith("…") else t[:-3].rstrip()
            # Avoid doubling if body starts with same prefix
            if not first_line.lower().startswith(t2.lower()):
                return f"{t2} {first_line}"
            return first_line
    return t


def sanitize_for_multimodal_sentinel(prompt: str) -> str:
    # Avoid DeepEval multimodal marker from being interpreted inside plain text prompts.
    return prompt.replace("[DEEPEVAL:IMAGE:", "[DEEPEVAL:IMG:")


######################
# GitHub API helpers #
######################


def gh_get(
    url: str, *, accept: Optional[str] = None, timeout_s: int = 20
) -> bytes:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "deepeval-changelog-generator")
    if accept:
        req.add_header("Accept", accept)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def gh_request(path: str, timeout_s: int = 20) -> dict:
    data = gh_get(
        f"https://api.github.com{path}",
        accept="application/vnd.github+json",
        timeout_s=timeout_s,
    )
    return json.loads(data.decode("utf-8"))


def fetch_pr(pr_number: int) -> Pull:
    data = gh_request(f"/repos/{OWNER}/{REPO}/pulls/{pr_number}")
    user_data = data.get("user") or {}
    return Pull(
        number=pr_number,
        title=data.get("title") or "",
        body=data.get("body") or "",
        merged_at=data.get("merged_at") or "",
        html_url=data.get("html_url")
        or f"https://github.com/{OWNER}/{REPO}/pull/{pr_number}",
        user_login=user_data.get("login") or "",
        user_html_url=user_data.get("html_url") or "",
        diff_url=data.get("diff_url") or "",
    )


def fetch_pr_diff(diff_url: str, timeout_s: int = 20) -> str:
    data = gh_get(
        diff_url, accept="application/vnd.github.v3.diff", timeout_s=timeout_s
    )
    return data.decode("utf-8", errors="replace")


def fetch_user_display(login: str) -> Tuple[str, str]:
    """
    Returns (display_name, html_url). display_name falls back to login.
    Cached per-login to avoid repeated requests.
    """
    login = (login or "").strip()
    if not login:
        return "", ""
    if login in user_cache:
        return user_cache[login]

    data = gh_request(f"/users/{login}")
    name = (data.get("name") or "").strip()
    html_url = (data.get("html_url") or "").strip()
    display = name or login
    user_cache[login] = (display, html_url)
    return user_cache[login]


###############
# LLM Helpers #
###############


def get_ai_model(model_name: str):
    from deepeval.models import GPTModel

    return GPTModel(model=model_name)


def build_ai_prompt(*, title: str, body: str) -> str:
    # Keep the instructions short + strict; rely on the schema for structure.
    return f"""
You are writing release notes for an open-source Python developer tool.

Task:
Given a PR title and PR body, produce:
- entry: one short, ClickHouse-style release note line (no markdown, no PR refs, no URLs)
- category: choose the best match from the allowed categories

Style rules (very important):
- Focus on the user-visible change and outcome.
- Use plain language; avoid internal jargon, code names, branch names, and "merge pull request".
- Prefer an action verb: "Add", "Fix", "Improve", "Reduce", "Prevent", "Support".
- Keep it to 1-4 sentences, plain text, target 120-500 chars not exceeding 500.
- If PR body provides enough detail, write 2-4 sentences. Otherwise keep to 1 sentence.
- Don’t mention "DeepEval" unless it is essential for clarity. Use your existing confidence to decide if you should fall back to title-only.
- If PR body is empty, write a single sentence based on title only.
- No version numbers, no PR numbers.
- You may use backticks for inline code (like_this) when appropriate.
- Do not use any other markdown (no lists, headers, links).

If the PR is unclear, write the safest high-level improvement without guessing details.
IMPORTANT: Output only valid JSON with no code fences or comments.

Allowed categories:
- Backward Incompatible Change
- New Feature
- Experimental Feature
- Improvement
- Bug Fix
- Security

PR title:
{title.strip()}

PR body (may include templates/checklists):
{(body or "").strip()}
""".strip()


def build_month_summary_prompt(*, month: str, entries: list[str]) -> str:
    # entries are your bullet texts (ideally without the PR link tail)
    joined = "\n".join(f"- {e}" for e in entries)
    return f"""
You are writing a short monthly release summary for an open-source Python developer tool.

Write 2–5 sentences of prose summarizing the themes and highlights for the month.
- No lists, no headings, no links, no PR numbers.
- Plain text.
- You may use backticks for inline code identifiers when appropriate.

Month:
{month}

Release note entries:
{joined}

IMPORTANT: Output only valid JSON.
""".strip()


def ai_month_summary(model, *, month: str, entries: list[str]) -> str:
    compact = entries[:80]
    prompt = sanitize_for_multimodal_sentinel(
        build_month_summary_prompt(month=month, entries=compact)
    )
    parsed, _cost = model.generate(prompt, schema=AiMonthSummary)
    assert isinstance(parsed, AiMonthSummary)
    return parsed.summary.strip()


def ai_release_note_for_pr(
    model,
    *,
    pr_number: int,
    title: str,
    body: str,
) -> tuple[AiReleaseNote, float]:
    prompt = sanitize_for_multimodal_sentinel(
        build_ai_prompt(title=title, body=body)
    )
    try:
        parsed, cost = model.generate(prompt, schema=AiReleaseNote)
        # GPTModel returns (BaseModel, cost) when schema is provided
        assert isinstance(parsed, AiReleaseNote)
    except Exception as e:
        raise RuntimeError(
            f"--ai failed for PR #{pr_number}. "
            f"Title={title!r}. Error={type(e).__name__}: {e}"
        ) from e
    return parsed, cost


def clean_pr_body_for_ai(body: str, *, max_chars: int = 2000) -> str:
    if not body:
        return ""

    s = body

    # Remove HTML comments (often template hints)
    s = re.sub(r"(?s)<!--.*?-->", "", s)

    # Remove <details> blocks (often long checklists / screenshots)
    s = re.sub(r"(?is)<details.*?>.*?</details>", "", s)

    lines: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()

        if not line:
            continue

        # Drop common checklist/template noise
        if re.match(r"^-\s*\[[ xX]\]\s+", line):
            continue
        if re.match(
            r"^(##|###)\s*(Checklist|Changelog|Testing|Test Plan|Screenshots|Notes)\b",
            line,
            re.I,
        ):
            continue
        if re.match(r"^(Closes|Fixes|Resolves)\s+#\d+", line, re.I):
            continue

        # Drop link dumps
        if re.match(r"^https?://\S+$", line):
            continue

        lines.append(line)

    out = "\n".join(lines).strip()

    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n\n[TRUNCATED]"
    return out


def clean_diff_for_ai(diff_text: str) -> str:
    """
    Light cleanup to make diffs more model-friendly, then truncate.

    - Drops very large/binary-ish sections (e.g., 'GIT binary patch').
    - Removes extremely long lines (often minified / generated).
    """
    if not diff_text:
        return ""

    lines: list[str] = []
    for ln in diff_text.splitlines():
        # Skip binary patches / obvious noise
        if "GIT binary patch" in ln:
            continue
        if ln.startswith("Binary files "):
            continue

        # Drop absurdly long lines (minified/compiled)
        if len(ln) > 2000:
            lines.append(ln[:2000] + " [LINE TRUNCATED]")
            continue

        lines.append(ln)

    cleaned = "\n".join(lines).strip()
    max_chars = AI_MAX_DIFF_LENGTH
    return truncate_text(
        cleaned,
        max_chars=max_chars,
        head_chars=int(max_chars * 0.6),
        tail_chars=int(max_chars * 0.25),
        marker="\n\n[... DIFF TRUNCATED ...]\n\n",
    )


#############
# Utilities #
#############


def truncate_text(
    text: str,
    *,
    max_chars: int = 12000,
    head_chars: int = 6000,
    tail_chars: int = 3000,
    marker: str = "\n\n[... TRUNCATED ...]\n\n",
) -> str:
    """
    Truncate large text safely.

    - If <= max_chars: return as is.
    - Otherwise: keep head_chars plus tail_chars with a marker between.
    """
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    # Ensure sane values
    head_chars = max(0, min(head_chars, max_chars))
    tail_chars = max(0, min(tail_chars, max_chars - head_chars))
    if head_chars == 0 and tail_chars == 0:
        return marker.strip()

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    return f"{head}{marker}{tail}".strip()


def strip_entry_tail(line: str) -> str:
    s = line.strip()
    if s.startswith("- "):
        s = s[2:]
    s = BULLET_TAIL_RE.sub("", s).strip()
    return s


#################################
# Classification / sanitization #
#################################


def clean_title(title: str) -> str:
    title = title.strip()
    title = re.sub(
        r"^(feat|fix|docs|perf|refactor|ci|chore)(\([^)]+\))?:\s*",
        "",
        title,
        flags=re.I,
    )
    return title.strip()


def classify(title: str, body: str) -> str:
    title_lower = title.lower()
    body_lower = (body or "").lower()

    if any(
        key_word in title_lower or key_word in body_lower
        for key_word in [
            "breaking",
            "backward incompatible",
            "incompatible",
            "breaking change",
        ]
    ):
        return "Backward Incompatible Change"
    if any(
        key_word in title_lower or key_word in body_lower
        for key_word in ["security", "vuln", "cve"]
    ):
        return "Security"
    if any(
        key_word in title_lower or key_word in body_lower
        for key_word in [
            "poc",
            "prototype",
            "spike",
            "experimental",
            "preview",
            "beta",
        ]
    ):
        return "Experimental Feature"
    if any(
        key_word in title_lower or key_word in body_lower
        for key_word in [
            "fix",
            "bug",
            "crash",
            "regression",
            "error",
            "fails",
            "failure",
        ]
    ):
        return "Bug Fix"
    if any(
        key_word in title_lower or key_word in body_lower
        for key_word in [
            "feat",
            "add",
            "introduce",
            "support",
            "enable",
            "flag",
            "option",
            "new",
        ]
    ):
        return "New Feature"
    return "Improvement"


def mdx_escape(s: str) -> str:
    # Prevent MDX JSX parsing issues
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("{", "\\{").replace("}", "\\}")
    return s


############################
# File parsing / rendering #
############################


def split_prefix_and_body(text: str) -> Tuple[str, str]:
    """
    Return (prefix_with_marker, body_after_marker).

    - The prefix includes any YAML frontmatter (the leading `--- ... ---` block),
      plus the `START_MARKER` line.
    - If an ignore block is present immediately after the marker, we keep it in the
      prefix as well so it won't be interpreted as changelog bullets.

    If the marker is missing, we preserve frontmatter (if present) and inject the
    marker into the prefix.
    """

    def _pull_top_ignore_block(s: str) -> Tuple[str, str]:
        s2 = s.lstrip("\n")
        matched = IGNORE_BLOCK_TOP_RE.match(s2)
        if not matched:
            return "", s
        ignore_block = s2[: matched.end()]
        rest = s2[matched.end() :]
        return ignore_block.rstrip("\n") + "\n", rest

    if START_MARKER in text:
        before, _, after = text.partition(START_MARKER)
        ignore_block, rest = _pull_top_ignore_block(after)
        prefix = before.rstrip() + "\n\n" + START_MARKER + "\n"
        if ignore_block:
            prefix += ignore_block
        body = rest.lstrip("\n")
        return prefix, body

    # Try to keep frontmatter if present
    if text.startswith("---"):
        matched = re.match(r"^---\n.*?\n---\n", text, flags=re.S)
        if matched:
            front = matched.group(0).rstrip()
            rest = text[matched.end() :]
            ignore_block, rest2 = _pull_top_ignore_block(rest)
            prefix = front + "\n\n" + START_MARKER + "\n"
            if ignore_block:
                prefix += ignore_block
            return prefix, rest2.lstrip("\n")

    # No frontmatter, just inject marker at top
    ignore_block, rest = _pull_top_ignore_block(text)
    prefix = START_MARKER + "\n"
    if ignore_block:
        prefix += ignore_block
    return prefix, rest.lstrip("\n")


def parse_ignore_prs(text: str) -> set[int]:
    """
    Parse PR numbers from one or more `<!-- changelog-ignore: ... -->` HTML comment blocks.

    Should be placed immediately after the `START_MARKER`, for example:

        <!-- changelog-ignore:
        - 1234
        - 5678
        -->

    Lines may contain comments which can be used to document why a PR is being ignored
    Any integers found in the block are treated as PR numbers.
    """
    ignored: set[int] = set()
    for matched in IGNORE_BLOCK_ANY_RE.finditer(text):
        block = matched.group(1)
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for pr_num in re.findall(r"\b\d+\b", line):
                try:
                    ignored.add(int(pr_num))
                except ValueError:
                    pass
    return ignored


def prune_ignored(idx: ChangelogIndex, ignore_prs: set[int]) -> int:
    """
    Remove any PR entries whose number is in `ignore_prs`.

    This is what makes deletions persist accross updates: add the PR number to the ignore block, re-run
    the generator, and the entry will be removed and it won't be re-added by future generator updates.
    """
    removed = 0
    for month, categories in list(idx.items()):
        for category, versions in list(categories.items()):
            for version, prs in list(versions.items()):
                for pr in list(prs.keys()):
                    if pr in ignore_prs:
                        del prs[pr]
                        removed += 1
    return removed


def parse_body(body: str) -> ChangelogIndex:
    idx: ChangelogIndex = {}
    month = None
    category = None
    version = None

    for line in body.splitlines():
        matched = MONTH_RE.match(line)
        if matched:
            month = matched.group(1).strip()
            idx.setdefault(month, {})
            category = None
            version = None
            continue
        matched = CATEGORY_RE.match(line)
        if matched:
            category = matched.group(1).strip()
            if month is None:
                continue
            idx[month].setdefault(category, {})
            version = None
            continue
        matched = VERSION_RE.match(line)
        if matched:
            version = matched.group(1).strip()
            if month is None or category is None:
                continue
            idx[month][category].setdefault(version, {})
            continue

        if line.startswith("- "):
            if month is None or category is None or version is None:
                continue
            matched = BULLET_PR_RE.search(line) or BULLET_PR_MARKER_RE.search(
                line
            )
            if not matched:
                continue
            pr = int(matched.group(1))
            idx[month][category][version][pr] = line.rstrip()

    return idx


def month_sort_key(name: str) -> int:
    return MONTH_INDEX.get(name, 0)


def render_changelog_body(
    idx: ChangelogIndex,
    version_date: Dict[str, str],
    *,
    use_ai: bool = False,
    ai_model: str = "gpt-5.2",
) -> str:
    """
    Render an ChangelogIndex into an MDX/Markdown changelog body.

    Output structure:
      - "## {Month}" sections (newest month first)
      - "### {Category}" subsections in CATEGORY_ORDER. Empty categories are omitted
      - "#### {Version}" blocks ordered by version_date desc
      - bullet entries under each version, sorted by PR number

    Returns the rendered body text with a trailing newline.
    """
    months = sorted(idx.keys(), key=month_sort_key, reverse=True)

    out: List[str] = []
    ai = get_ai_model(ai_model) if use_ai else None
    for month in months:
        out.append(f"## {month}")
        out.append("")

        # Monthly summary
        if use_ai and ai is not None:
            month_entries: list[str] = []
            for _category, versions in idx[month].items():
                for _version, prs in versions.items():
                    for _pr, line in prs.items():
                        month_entries.append(strip_entry_tail(line))

            if month_entries:
                try:
                    summary = ai_month_summary(
                        ai, month=month, entries=month_entries
                    )
                    summary = mdx_escape(summary)
                    out.append(summary)
                except Exception as e:
                    # Don't kill changelog rendering if summary fails
                    print(f"[month summary] {month}: {type(e).__name__}: {e}")
                out.append("")

        for category in CATEGORY_ORDER:
            if category not in idx[month]:
                continue
            # only render those that actually have entries
            has_any = any(idx[month][category].values())
            if not has_any:
                continue

            out.append(f"### {category}")
            out.append("")

            # version DESC by tag date
            versions = list(idx[month][category].keys())
            versions.sort(key=lambda v: version_date.get(v, ""), reverse=True)

            for version in versions:
                entries = idx[month][category][version]
                if not entries:
                    continue
                out.append(f"#### {version}")
                # ascending by PR number
                for pr in sorted(entries.keys()):
                    out.append(entries[pr])
                out.append("")  # blank line after each version block

            out.append("")  # blank line after category

        out.append("")  # blank line after month

    # Trim trailing blank lines
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + ("\n" if out else "")


###################
# Build and merge #
###################

JUNK_TITLE_RE = re.compile(
    r"^(merge pull request|merge branch|bump |release |main$|master$|patch-\d+|hotfix|wip)\b",
    re.IGNORECASE,
)


def title_needs_github(title: str) -> bool:
    title = (title or "").strip()
    if not title:
        return True
    if MERGE_SUBJECT_RE.match(title):
        return True
    if JUNK_TITLE_RE.match(title):
        return True
    if re.fullmatch(r"[\w.-]+/[\w.-]+", title):
        return True
    return False


def month_name_from_ymd(ymd: str) -> Tuple[int, str]:
    year, month, _ = map(int, ymd.split("-"))
    return year, MONTH_NAMES[month - 1]


def build_release_entries(
    tag: str,
    use_github: bool,
    use_ai: bool = False,
    ai_model: str = "gpt-5.2",
    sleep_s: float = 0.0,
    ignore_prs: Optional[set[int]] = None,
    existing_keys: Optional[set[tuple[str, int]]] = None,
    overwrite_existing: bool = False,
    status_cb: Optional[Callable[[str], None]] = None,
    tick_cb: Optional[Callable[[], None]] = None,
) -> Tuple[int, str, ChangelogIndex, Dict[str, str], float]:
    prev = get_prev_tag(tag)
    tag_date = git_tag_date_ymd(tag)
    year, month = month_name_from_ymd(tag_date)

    commits = commits_in_range(prev, tag)
    pr_map = extract_pr_numbers(commits)
    if ignore_prs:
        pr_map = {
            pr: commit for pr, commit in pr_map.items() if pr not in ignore_prs
        }

    # collect entries for this tag into an index shape
    idx: ChangelogIndex = {month: {}}
    version_date = {tag: tag_date}
    ai = None
    ai_cache: dict[int, AiReleaseNote] = {}
    ai_total_cost = 0.0
    if use_ai:
        ai = get_ai_model(ai_model)

    def _status(msg: str) -> None:
        if status_cb is not None:
            status_cb(msg)

    def _tick() -> None:
        if tick_cb:
            tick_cb()

    for pr_num, commit in sorted(pr_map.items(), key=lambda kv: kv[0]):
        _status(f"[{tag}] PR #{pr_num}: preparing…")
        key = (tag, pr_num)
        if existing_keys and (key in existing_keys) and not overwrite_existing:
            _status(f"[{tag}] PR #{pr_num}: skipping (already present)")
            _tick()
            # Preserve manual edits/moves and avoid useless LLM calls
            continue

        # offline title from merge commit body if possible
        title = offline_pr_title_from_merge_commit(commit.sha, commit.subject)
        body = ""
        user_login = ""
        user_html_url = ""
        user_display = ""
        user_profile_url = ""
        diff_url = ""

        if use_github and (use_ai or title_needs_github(title)):
            _status(f"[{tag}] PR #{pr_num}: fetching from GitHub…")
            try:
                pr = fetch_pr(pr_num)
                diff_url = pr.diff_url
            except urllib.error.HTTPError as e:
                msg = (
                    f"Unable to fetch PR #{pr_num} for tag {tag} (commit {commit.sha[:8]}): "
                    f"HTTP {e.code} {e.reason}"
                )
                _status(f"[{tag}] PR #{pr_num}: error: HTTP {e.code}")
                print(msg)
                if e.code == 404:
                    _status(f"[{tag}] PR #{pr_num}: 404 (skipped)")
                    _tick()
                    continue
                raise
            except Exception as e:
                msg = (
                    f"Unable to fetch PR #{pr_num} for tag {tag} (commit {commit.sha[:8]}): "
                    f"{type(e).__name__}: {e}"
                )
                _status(f"[{tag}] PR #{pr_num}: error: {type(e).__name__}")
                print(msg)
                raise

            title = pr.title or title
            body = pr.body or ""
            if sleep_s:
                time.sleep(sleep_s)
            user_login = pr.user_login
            user_html_url = pr.user_html_url
            user_display, user_profile_url = fetch_user_display(user_login)
            # prefer profile url from user endpoint if present
            user_profile_url = user_profile_url or user_html_url

        body_clean = clean_pr_body_for_ai(body)
        has_detail = len(body_clean) >= 200
        title = stitch_truncated_title(title, body_clean)

        diff = ""
        if use_ai and use_github and (not has_detail) and diff_url:
            try:
                _status(f"[{tag}] PR #{pr_num}: fetching diff…")
                diff = fetch_pr_diff(diff_url)
                diff = clean_diff_for_ai(diff)
            except Exception:
                diff = ""

        # Use AI to generate a higher-quality bullet.
        if use_ai:
            if pr_num in ai_cache:
                note = ai_cache[pr_num]
            else:
                body_for_ai = body_clean if has_detail else ""
                if diff:
                    body_for_ai = (
                        (body_for_ai + "\n\n" if body_for_ai else "")
                        + "PR diff (for context; may be truncated):\n"
                        + diff
                    )

                note, cost = ai_release_note_for_pr(
                    ai,
                    pr_number=pr_num,
                    title=title,
                    body=body_for_ai,
                )
                ai_cache[pr_num] = note
                ai_total_cost += cost if cost is not None else 0

            bullet = mdx_escape(clean_title(note.entry.strip()))
            if not bullet.endswith("."):
                bullet += "."
            category = note.category
            title_out = bullet
        else:
            title_out = mdx_escape(clean_title(title))
            if not title_out.endswith("."):
                title_out += "."
            category = classify(title, body)

        idx[month].setdefault(category, {}).setdefault(tag, {})
        author = ""
        if user_display:
            if user_profile_url:
                author = f" ([{user_display}]({user_profile_url}))"
            else:
                author = f" ({user_display})"
        line = (
            f"- {title_out} ([#{pr_num}](https://github.com/{OWNER}/{REPO}/pull/{pr_num})) "
            f"<!-- pr:{pr_num} -->{author}"
        )
        idx[month][category][tag][pr_num] = line
        _status(f"[{tag}] PR #{pr_num}: done")
        _tick()
    return year, month, idx, version_date, ai_total_cost


def collect_existing_keys(idx: ChangelogIndex) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for _month, categories in idx.items():
        for _category, versions in categories.items():
            for version, prs in versions.items():
                for pr in prs.keys():
                    out.add((version, pr))
    return out


def merge_idx(
    existing: ChangelogIndex,
    updates: ChangelogIndex,
    overwrite_existing: bool = False,
) -> int:
    """Merge `updates` entries into `existing` (in-place).

    Entries are keyed by PR number and version tag. If an entry for the same
    (version,PR) already exists anywhere in `existing`, that location is treated
    as the correct location so manual moves between categories and months persist across
    updates.

    If `overwrite_existing` is False, existing entries are left untouched.
    If True, the existing bullet line is updated in-place.

    Returns the number of newly-added entries."""
    added = 0

    # Build a quick lookup of where each (version, PR) currently lives.
    loc_by_key: Dict[Tuple[str, int], Tuple[str, str]] = {}
    for month, categories in existing.items():
        for category, versions in categories.items():
            for version, prs in versions.items():
                for pr in prs.keys():
                    loc_by_key[(version, pr)] = (month, category)

    for month, categories in updates.items():
        existing.setdefault(month, {})
        for category, versions in categories.items():
            existing[month].setdefault(category, {})
            for version, prs in versions.items():
                existing[month][category].setdefault(version, {})
                for pr, line in prs.items():
                    key = (version, pr)
                    if key in loc_by_key:
                        month0, category0 = loc_by_key[key]
                        if not overwrite_existing:
                            continue
                        existing[month0].setdefault(category0, {})
                        existing[month0][category0].setdefault(version, {})
                        existing[month0][category0][version][pr] = line
                        continue

                    # new entry
                    added += 1
                    existing[month][category][version][pr] = line
                    loc_by_key[key] = (month, category)

    return added


def run_with_overall_progress(
    tags: list[str],
    args,
    per_year,
    per_year_prefix,
    per_year_ignore,
    version_date_entries,
) -> float:
    ai_total_cost = 0.0

    # If --silent, skip all rich UI and just run normally.
    if args.silent:
        for tag in tags:
            y, _m = month_name_from_ymd(git_tag_date_ymd(tag))
            out_path = os.path.join(args.output_dir, f"changelog-{y}.mdx")

            if y not in per_year:
                if os.path.exists(out_path):
                    existing_text = open(out_path, "r", encoding="utf-8").read()
                    prefix, body = split_prefix_and_body(existing_text)
                    per_year_prefix[y] = prefix
                    per_year_ignore[y] = parse_ignore_prs(prefix)
                    per_year[y] = parse_body(body)
                    for _month, categories in per_year[y].items():
                        for _cat, versions in categories.items():
                            for version in versions.keys():
                                if version not in version_date_entries:
                                    version_date_entries[version] = (
                                        git_tag_date_ymd(version)
                                    )
                else:
                    os.makedirs(args.output_dir, exist_ok=True)
                    per_year_prefix[y] = (
                        f"---\n"
                        f"id: changelog-{y}\n"
                        f"title: {y}\n"
                        f"sidebar_label: {y}\n"
                        f"---\n\n"
                        f"{START_MARKER}\n"
                    )
                    per_year_ignore[y] = set()
                    per_year[y] = {}

            existing_keys = collect_existing_keys(per_year[y])
            year, month, idx_update, vd, ai_cost = build_release_entries(
                tag,
                use_github=args.github,
                use_ai=args.ai,
                ai_model=args.ai_model,
                sleep_s=args.sleep,
                ignore_prs=per_year_ignore[y],
                existing_keys=existing_keys,
                overwrite_existing=args.overwrite_existing,
            )
            ai_total_cost += ai_cost
            version_date_entries.update(vd)
            merge_idx(
                per_year[year],
                idx_update,
                overwrite_existing=args.overwrite_existing,
            )

        return ai_total_cost

    console = Console(stderr=True)

    overall = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=26),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=CLEAR_PROGRESS_BAR_ON_COMPLETION,
        console=console,
    )

    # We would like the per tag progress indicator to remain only if it fails to cmplete due to an error, other wise we would like it to be removed at the end of the run.
    # The Key trick to getting the behavior we want is to make this transient=False like the overall indicator, but remove tasks on success.
    # That way, if there’s an error, the last per tag line stays visible.
    per_tag = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    )

    with Live(Group(overall, per_tag), console=console, refresh_per_second=10):
        overall_task = overall.add_task("Processing releases…", total=len(tags))

        for tag in tags:
            tag_task = per_tag.add_task(f"{tag}: preparing…", total=0)
            # Determine the output year early so we can load existing content and the ignore list
            y, _m = month_name_from_ymd(git_tag_date_ymd(tag))
            out_path = os.path.join(args.output_dir, f"changelog-{y}.mdx")

            if y not in per_year:
                if os.path.exists(out_path):
                    existing_text = open(out_path, "r", encoding="utf-8").read()
                    prefix, body = split_prefix_and_body(existing_text)
                    per_year_prefix[y] = prefix
                    per_year_ignore[y] = parse_ignore_prs(
                        prefix
                    )  # ignore block is kept in prefix
                    per_year[y] = parse_body(body)
                    for _month, categories in per_year[y].items():
                        for _cat, versions in categories.items():
                            for version in versions.keys():
                                if version not in version_date_entries:
                                    version_date_entries[version] = (
                                        git_tag_date_ymd(version)
                                    )
                else:
                    os.makedirs(args.output_dir, exist_ok=True)
                    per_year_prefix[y] = (
                        f"---\n"
                        f"id: changelog-{y}\n"
                        f"title: {y}\n"
                        f"sidebar_label: {y}\n"
                        f"---\n\n"
                        f"{START_MARKER}\n"
                    )
                    per_year_ignore[y] = set()
                    per_year[y] = {}

            existing_keys = collect_existing_keys(per_year[y])

            # Compute total PRs we expect to handle for this tag so the bar is the right length.
            prev = get_prev_tag(tag)
            commits = commits_in_range(prev, tag)
            pr_map = extract_pr_numbers(commits)
            if per_year_ignore.get(y):
                pr_map = {
                    pr: c
                    for pr, c in pr_map.items()
                    if pr not in per_year_ignore[y]
                }
            per_tag.update(tag_task, total=len(pr_map), completed=0)

            def status_cb(msg: str) -> None:
                per_tag.update(tag_task, description=escape(msg))

            def tick_cb() -> None:
                per_tag.advance(tag_task, 1)

            per_tag.update(tag_task, description=f"{tag}: generating entries…")
            try:
                year, month, idx_update, vd, ai_cost = build_release_entries(
                    tag,
                    use_github=args.github,
                    use_ai=args.ai,
                    ai_model=args.ai_model,
                    sleep_s=args.sleep,
                    ignore_prs=per_year_ignore[y],
                    existing_keys=existing_keys,
                    overwrite_existing=args.overwrite_existing,
                    status_cb=status_cb,
                    tick_cb=tick_cb,
                )
            except Exception as e:
                # Leave the tag line visible (do NOT remove task), show concise error.
                per_tag.update(
                    tag_task,
                    description=f"{tag}: error: {type(e).__name__}: {e}",
                )
                raise
            else:
                ai_total_cost += ai_cost
                version_date_entries.update(vd)
                merge_idx(
                    per_year[year],
                    idx_update,
                    overwrite_existing=args.overwrite_existing,
                )

                per_tag.update(tag_task, description=f"{tag}: done")
                per_tag.remove_task(tag_task)  # remove on success
                overall.advance(overall_task, 1)

    return ai_total_cost


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="Release tag like v3.7.6")
    g.add_argument(
        "--latest", action="store_true", help="Generate for latest tag only"
    )
    g.add_argument(
        "--range",
        nargs=2,
        metavar=("FROM", "TO"),
        help="Generate for an inclusive tag range",
    )
    g.add_argument(
        "--year",
        type=int,
        help="Generate for all tags whose tag date falls within YEAR",
    )

    ap.add_argument(
        "--output-dir", default="docs/changelog", help="Docs changelog dir"
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="Enrich titles/bodies from GitHub API (needs token for speed)",
    )
    ap.add_argument(
        "--ai",
        action="store_true",
        help="Use an LLM to generate release-note bullets",
    )
    ap.add_argument("--ai-model", default="gpt-5.2", help="Model name for --ai")
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing entries for the same PR (default preserves manual edits)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between GitHub API calls (seconds)",
    )
    ap.add_argument(
        "--silent",
        action="store_true",
        help="Disable progress indicator output.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ai_total_cost = 0.0
    if args.latest:
        tags = [latest_tag()]
    elif args.tag:
        tags = [args.tag]
    elif args.year is not None:
        tags = list_tags_for_year(args.year)
    else:
        tags = list_tags_between(args.range[0], args.range[1])

    # Load existing year files into memory, merge updates, then write once per year.
    per_year: Dict[int, ChangelogIndex] = {}
    per_year_prefix: Dict[int, str] = {}
    version_date_entries: Dict[str, str] = {}
    per_year_ignore: Dict[int, set[int]] = {}

    ai_total_cost = run_with_overall_progress(
        tags,
        args,
        per_year,
        per_year_prefix,
        per_year_ignore,
        version_date_entries,
    )

    # Write outputs
    for year, idx in per_year.items():
        prune_ignored(idx, per_year_ignore.get(year, set()))
        out_path = os.path.join(args.output_dir, f"changelog-{year}.mdx")
        body = render_changelog_body(
            idx,
            version_date=version_date_entries,
            use_ai=args.ai,
            ai_model=args.ai_model,
        )
        text = per_year_prefix[year].rstrip() + "\n\n" + body

        if args.dry_run:
            print(f"Would write {out_path}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Wrote {out_path}")

    if args.ai:
        print(f"AI total cost: ${ai_total_cost:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
