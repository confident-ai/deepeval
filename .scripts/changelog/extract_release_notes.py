#!/usr/bin/env python3
import argparse
import os
import sys
import re

# Import parsing helpers from generate.py
try:
    from generate import split_prefix_and_body, parse_body, CATEGORY_ORDER
except ImportError:
    # Add current directory to path if run directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from generate import split_prefix_and_body, parse_body, CATEGORY_ORDER


def main():
    parser = argparse.ArgumentParser(
        description="Extract release notes for a specific tag from the generated changelog"
    )
    parser.add_argument(
        "--tag", required=True, help="Tag to extract notes for (e.g. v0.21.45)"
    )
    parser.add_argument(
        "--changelog-dir",
        default="docs/content/changelog",
        help="Directory containing changelog MDX files",
    )
    parser.add_argument(
        "--output",
        default="release_notes.md",
        help="Output file for the extracted notes",
    )
    args = parser.parse_args()

    if not os.path.exists(args.changelog_dir):
        print(f"Error: {args.changelog_dir} does not exist.")
        sys.exit(1)

    extracted_notes = {}
    contributors = set()

    # Iterate through all changelog files to find the tag
    for filename in os.listdir(args.changelog_dir):
        if not filename.endswith(".mdx"):
            continue

        filepath = os.path.join(args.changelog_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        _, body = split_prefix_and_body(text)
        idx = parse_body(body)

        # idx structure: month -> category -> version -> pr_number -> bullet_line
        for month, categories in idx.items():
            for category, versions in categories.items():
                if args.tag in versions:
                    if category not in extracted_notes:
                        extracted_notes[category] = []

                    # Sort PRs sequentially
                    prs = versions[args.tag]
                    for pr in sorted(prs.keys()):
                        line = prs[pr]

                        # Extract author (appended after the pr marker)
                        parts = line.split("*/}")
                        if len(parts) > 1:
                            author_part = parts[-1].strip()
                            if (
                                author_part
                                and author_part.startswith("(")
                                and author_part.endswith(")")
                            ):
                                author = author_part[1:-1]
                                contributors.add(author)

                        # Remove the PR marker for a cleaner release note
                        line = re.sub(r"\s*\{/\*\s*pr:\d+\s*\*/\}", "", line)

                        extracted_notes[category].append(line)

    if not extracted_notes:
        print(f"No release notes found for tag {args.tag}")
        # Create an empty file so the workflow knows fallback is needed
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("")
        sys.exit(0)

    # Format the extracted notes into markdown
    out_lines = []

    out_lines.append("## Getting started with deepeval? Run:")
    out_lines.append("```bash")
    out_lines.append("pip install deepeval")
    out_lines.append("```")

    # Sort categories based on the order defined in generate.py
    for category in CATEGORY_ORDER:
        if category in extracted_notes:
            out_lines.append(f"### {category}")
            out_lines.append("")
            for line in extracted_notes[category]:
                out_lines.append(line)
            out_lines.append("")  # blank line for spacing

    if contributors:
        formatted_contributors = []
        for c in sorted(contributors):
            # Try to extract the GitHub username to ping them using @
            match = re.match(r"\[.+\]\(https://github\.com/([^/]+)\)", c)
            if match:
                formatted_contributors.append(f"@{match.group(1)}")
            else:
                formatted_contributors.append(c)

        out_lines.append("### ❤️ Contributors")
        out_lines.append("")
        out_lines.append(
            "A huge thank you to our contributors for this release: "
            + ", ".join(formatted_contributors)
        )
        out_lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines).strip() + "\n")

    print(
        f"Successfully extracted release notes for {args.tag} to {args.output}"
    )


if __name__ == "__main__":
    main()
