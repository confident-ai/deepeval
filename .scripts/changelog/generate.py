#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Commit:
    sha: str
    subject: str


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

PR_NUM_RE = re.compile(r"\(#(\d+)\)|pull request #(\d+)", re.IGNORECASE)
MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #(\d+)\b", re.IGNORECASE)

###############
# Git helpers #
###############


def sh(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def git_tag_date_ymd(tag: str) -> str:
    return sh(["git", "log", "-1", "--format=%cs", tag])


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


def latest_tag() -> str:
    return sh(["git", "describe", "--tags", "--abbrev=0", "--match", "v*"])


def commits_in_range(base: str, head: str) -> List[Commit]:
    # get sha and subject for commit subjects in range
    raw = sh(["git", "log", "--format=%H%x00%s", f"{base}..{head}"])
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


######################
# GitHub API helpers #
######################


def gh_request(path: str, timeout_s: int = 20) -> dict:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "deepeval-changelog-generator")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class Pull:
    number: int
    title: str
    body: str
    merged_at: str
    html_url: str


def fetch_pr(pr_number: int) -> Pull:
    data = gh_request(f"/repos/{OWNER}/{REPO}/pulls/{pr_number}")
    return Pull(
        number=pr_number,
        title=data.get("title") or "",
        body=data.get("body") or "",
        merged_at=data.get("merged_at") or "",
        html_url=data.get("html_url")
        or f"https://github.com/{OWNER}/{REPO}/pull/{pr_number}",
    )


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

EntryIndex = Dict[str, Dict[str, Dict[str, Dict[int, str]]]]
# month -> category -> version -> pr_number -> bullet_line

MONTH_RE = re.compile(r"^##\s+(.+?)\s*$")
CATEGORY_RE = re.compile(r"^###\s+(.+?)\s*$")
VERSION_RE = re.compile(r"^####\s+(v[0-9].+?)\s*$")
BULLET_PR_RE = re.compile(r"\[#(\d+)\]\(")


def split_prefix_and_body(text: str) -> Tuple[str, str]:
    """
    Split an MDX file into:
      - prefix: everything up to and including START_MARKER
      - body: everything after START_MARKER

    START_MARKER is a sentinel comment like that separates the generated changelog
    region from the file header.

    If START_MARKER is missing:
      - If the file begins with YAML frontmatter (a top-of-file block delimited by
        '---' lines and containing Docusaurus 'id/title/sidebar_label'), preserve that block
        and insert START_MARKER immediately after it.
      - Otherwise, insert START_MARKER at the top of the file.

    Returns (prefix_with_marker, body_after_marker).
    """
    if START_MARKER in text:
        before, _, after = text.partition(START_MARKER)
        prefix = before.rstrip() + "\n\n" + START_MARKER + "\n"
        body = after.lstrip("\n")
        return prefix, body

    # Try to keep frontmatter if present
    if text.startswith("---"):
        matched = re.match(r"^---\n.*?\n---\n", text, flags=re.S)
        if matched:
            front = matched.group(0).rstrip()
            rest = text[matched.end() :].lstrip("\n")
            prefix = front + "\n\n" + START_MARKER + "\n"
            return prefix, rest

    # No frontmatter, just inject marker at top
    prefix = START_MARKER + "\n"
    return prefix, text.lstrip("\n")


def parse_body(body: str) -> EntryIndex:
    idx: EntryIndex = {}
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
            matched = BULLET_PR_RE.search(line)
            if not matched:
                continue
            pr = int(matched.group(1))
            idx[month][category][version][pr] = line.rstrip()

    return idx


def month_sort_key(name: str) -> int:
    return MONTH_INDEX.get(name, 0)


def render_changelog_body(idx: EntryIndex, version_date: Dict[str, str]) -> str:
    """
    Render an EntryIndex into an MDX/Markdown changelog body.

    Output structure:
      - "## {Month}" sections (newest month first)
      - "### {Category}" subsections in CATEGORY_ORDER. Empty categories are omitted
      - "#### {Version}" blocks ordered by version_date desc
      - bullet entries under each version, sorted by PR number

    Returns the rendered body text with a trailing newline.
    """
    months = sorted(idx.keys(), key=month_sort_key, reverse=True)

    out: List[str] = []
    for month in months:
        out.append(f"## {month}")
        out.append("")

        # categories in fixed order, but only render those that actually have entries
        for category in CATEGORY_ORDER:
            if category not in idx[month]:
                continue
            # skip if empty
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
    tag: str, use_github: bool, sleep_s: float = 0.0
) -> Tuple[int, str, EntryIndex, Dict[str, str]]:
    prev = get_prev_tag(tag)
    tag_date = git_tag_date_ymd(tag)
    year, month = month_name_from_ymd(tag_date)

    commits = commits_in_range(prev, tag)
    pr_map = extract_pr_numbers(commits)

    # collect entries for this tag into an index shape
    idx: EntryIndex = {month: {}}
    version_date = {tag: tag_date}

    for pr_num, commit in sorted(pr_map.items(), key=lambda kv: kv[0]):
        # offline title from merge commit body if possible
        title = offline_pr_title_from_merge_commit(commit.sha, commit.subject)
        body = ""

        if use_github and title_needs_github(title):
            print(
                f"Fetching PR #{pr_num} from GitHub…",
                file=sys.stderr,
                flush=True,
            )
            pr = fetch_pr(pr_num)
            title = pr.title or title
            body = pr.body or ""
            if sleep_s:
                time.sleep(sleep_s)

        title = clean_title(title)
        title = mdx_escape(title)
        if not title.endswith("."):
            title += "."

        category = classify(title, body)

        idx[month].setdefault(category, {}).setdefault(tag, {})
        line = f"- {title} ([#{pr_num}](https://github.com/{OWNER}/{REPO}/pull/{pr_num}))"
        idx[month][category][tag][pr_num] = line

    return year, month, idx, version_date


def merge_idx(existing: EntryIndex, new_entries: EntryIndex) -> int:
    """
    Merge changelog entries from `new_entries` into `existing`.

    Performs a deep merge across month -> category -> version -> PR number.
    Buckets are created as needed. For each (month, category, version, pr) key:
      - If the PR is new to `existing`, increments the returned `added` count.
      - Always sets/overwrites the stored bullet line, allowing later runs
        (e.g., GitHub-enriched titles) to replace earlier text.

    Returns:
        The number of newly added PR entries (not counting overwrites).
    """
    added = 0
    for month, categories in new_entries.items():
        existing.setdefault(month, {})
        for category, versions in categories.items():
            existing[month].setdefault(category, {})
            for version, prs in versions.items():
                existing[month][category].setdefault(version, {})
                for pr, line in prs.items():
                    # same (month,category,version,pr) overwrites (lets github improve titles)
                    if pr not in existing[month][category][version]:
                        added += 1
                    existing[month][category][version][pr] = line
    return added


# ----------------------------
# CLI
# ----------------------------


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

    ap.add_argument(
        "--output-dir", default="docs/docs/changelog", help="Docs changelog dir"
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="Enrich titles/bodies from GitHub API (needs token for speed)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between GitHub API calls (seconds)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.latest:
        tags = [latest_tag()]
    elif args.tag:
        tags = [args.tag]
    else:
        tags = list_tags_between(args.range[0], args.range[1])

    # Load existing year files into memory, merge updates, then write once per year.
    per_year: Dict[int, EntryIndex] = {}
    per_year_prefix: Dict[int, str] = {}
    version_date: Dict[str, str] = {}

    for tag in tags:
        year, month, idx, version_date = build_release_entries(
            tag, use_github=args.github, sleep_s=args.sleep
        )
        version_date.update(version_date)

        out_path = os.path.join(args.output_dir, f"changelog-{year}.mdx")
        if year not in per_year:
            if os.path.exists(out_path):
                text = open(out_path, "r", encoding="utf-8").read()
                prefix, body = split_prefix_and_body(text)
                per_year_prefix[year] = prefix
                per_year[year] = parse_body(body)
            else:
                os.makedirs(args.output_dir, exist_ok=True)
                per_year_prefix[year] = (
                    f"---\n"
                    f"id: changelog-{year}\n"
                    f"title: {year}\n"
                    f"sidebar_label: {year}\n"
                    f"---\n\n"
                    f"{START_MARKER}\n"
                )
                per_year[year] = {}

        merge_idx(per_year[year], idx)

    # Write outputs
    for year, idx in per_year.items():
        out_path = os.path.join(args.output_dir, f"changelog-{year}.mdx")
        body = render_changelog_body(idx, version_date=version_date)
        text = per_year_prefix[year].rstrip() + "\n\n" + body

        if args.dry_run:
            print(f"Would write {out_path}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
