import io
import os

import time
from typing import List
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.terminal_theme import TerminalTheme

from deepeval.evaluate.types import TestResult

LIGHT_THEME = TerminalTheme(
    background=(0, 0, 0),
    foreground=(255, 255, 255),
    normal=[
        (0, 0, 0),
        (205, 49, 49),
        (13, 188, 121),
        (229, 229, 16),
        (36, 114, 200),
        (188, 63, 188),
        (17, 168, 205),
        (229, 229, 229),
    ],
    bright=[
        (102, 102, 102),
        (241, 76, 76),
        (35, 209, 139),
        (245, 245, 67),
        (59, 142, 234),
        (214, 112, 214),
        (41, 184, 219),
        (229, 229, 229),
    ],
)

DEEPEVAL_PURPLE = "rgb(106,0,255)"
DEEPEVAL_GREEN = "rgb(25,227,160)"
FAIL_RED = "red"

import re


def _natural_sort_key(s: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


class EvaluationConsoleReport:
    def __init__(self, test_results: List[TestResult]):
        self.test_results = sorted(
            test_results,
            key=lambda x: (
                x.index if x.index is not None else float("inf"),
                _natural_sort_key(x.name),
            ),
        )
        self.console = Console()

    def _build_display_elements(self, truncate: bool = True) -> Group:

        renderables = [
            Panel(
                f"[{DEEPEVAL_PURPLE} bold]🚀 DeepEval Evaluation Results[/{DEEPEVAL_PURPLE} bold]",
                expand=True,
            )
        ]

        for case in self.test_results:
            status_color = DEEPEVAL_GREEN if case.success else FAIL_RED
            status_icon = "✅" if case.success else "❌"

            if truncate and case.success:
                summary_text = f"[{status_color} bold]{status_icon} {case.name} (Passed {len(case.metrics_data)} metrics)[/{status_color} bold]"
                renderables.append(
                    Panel(summary_text, border_style=status_color, expand=True)
                )
                continue

            content_tree = Tree(
                f"[{status_color} bold]{status_icon} {case.name}[/{status_color} bold]"
            )

            if case.conversational:
                convo_tree = content_tree.add(
                    "[bold cyan]Conversation Turns[/bold cyan]"
                )
                for turn in case.turns:
                    convo_tree.add(
                        f"[bold]{turn.role.capitalize()}:[/bold] {turn.content}"
                    )
            else:
                data_table = Table(show_header=False, box=None, padding=(0, 2))
                data_table.add_column("Key", style="bold cyan")
                data_table.add_column("Value")
                data_table.add_row("Input:", str(case.input))
                data_table.add_row("Actual Output:", str(case.actual_output))
                if case.expected_output and case.expected_output != "N/A":
                    data_table.add_row(
                        "Expected Output:", str(case.expected_output)
                    )
                content_tree.add(data_table)

            metrics_table = Table(
                title="Metrics",
                title_justify="left",
                show_edge=False,
                header_style=f"bold {DEEPEVAL_PURPLE}",
                expand=True,
            )
            metrics_table.add_column("Status", justify="center")
            metrics_table.add_column("Metric")
            metrics_table.add_column("Score")
            metrics_table.add_column("Threshold")
            metrics_table.add_column("Reason")

            for m in case.metrics_data:
                m_icon = (
                    "[bold green]PASS[/bold green]"
                    if m.success
                    else "[bold red]FAIL[/bold red]"
                )
                if m.error:
                    m_icon = "[bold red]ERROR[/bold red]"

                score_str = f"{m.score:.2f}" if m.score is not None else "N/A"
                thresh_str = (
                    f"{m.threshold:.2f}" if m.threshold is not None else "N/A"
                )
                reason_str = str(m.reason or m.error or "N/A")

                if truncate and m.success and len(reason_str) > 50:
                    reason_str = reason_str[:47] + "..."

                metrics_table.add_row(
                    m_icon, m.name, score_str, thresh_str, reason_str
                )

            content_tree.add(metrics_table)
            renderables.append(
                Panel(
                    content_tree,
                    border_style=status_color,
                    padding=(1, 2),
                    expand=True,
                )
            )

        # Calculate aggregate metrics
        metric_aggregates = {}
        for case in self.test_results:
            for m in case.metrics_data:
                if m.name not in metric_aggregates:
                    metric_aggregates[m.name] = {
                        "total": 0,
                        "passes": 0,
                        "score_sum": 0,
                        "score_count": 0,
                    }

                agg = metric_aggregates[m.name]
                agg["total"] += 1
                if m.success:
                    agg["passes"] += 1
                if m.score is not None:
                    agg["score_sum"] += m.score
                    agg["score_count"] += 1

        if metric_aggregates:
            # Adding some padding below header
            agg_table = Table(
                title="[bold]Aggregate Metrics[/bold]\n",
                title_justify="left",
                show_edge=False,
                header_style=f"bold {DEEPEVAL_PURPLE}",
                expand=True,
            )
            agg_table.add_column("Metric")
            agg_table.add_column("Average Score")
            agg_table.add_column("Pass Rate")
            agg_table.add_column("Total")

            for metric_name, agg in metric_aggregates.items():
                avg_score = (
                    f"{agg['score_sum'] / agg['score_count']:.2f}"
                    if agg["score_count"] > 0
                    else "N/A"
                )
                pass_rate = (
                    f"{(agg['passes'] / agg['total']) * 100:.2f}%"
                    if agg["total"] > 0
                    else "N/A"
                )
                agg_table.add_row(
                    metric_name, avg_score, pass_rate, str(agg["total"])
                )

            renderables.append(
                Panel(agg_table, border_style=DEEPEVAL_PURPLE, expand=True)
            )

        return Group(*renderables)

    def render_to_terminal(self, truncate_passing_cases: bool = True):
        self.console.print()
        self.console.print(
            self._build_display_elements(truncate=truncate_passing_cases)
        )
        self.console.print()

    def export_to_html(
        self,
        output_dir: str,
        evaluation_name: str = "evaluation",
        theme_mode: str = "dark",
    ):
        os.makedirs(output_dir, exist_ok=True)

        safe_name = (
            str(evaluation_name).replace(" ", "_").lower()
            if evaluation_name
            else "evaluation"
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{safe_name}_{timestamp}.html")

        dummy_file = io.StringIO()
        html_console = Console(
            record=True, file=dummy_file, force_terminal=True
        )
        html_console.print(self._build_display_elements(truncate=False))

        html_console.save_html(filepath, theme=LIGHT_THEME)

        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()

        css_patch = "<style>pre { line-height: 1.1 !important; }</style></head>"
        html_content = html_content.replace("</head>", css_patch)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML Dashboard saved to: {filepath}")

    def export_to_markdown(
        self, output_dir: str, evaluation_name: str = "evaluation"
    ):
        os.makedirs(output_dir, exist_ok=True)

        safe_name = (
            str(evaluation_name).replace(" ", "_").lower()
            if evaluation_name
            else "evaluation"
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{safe_name}_{timestamp}.md")

        md = ["# 🚀 DeepEval Evaluation Results\n"]

        for case in self.test_results:
            status_icon = "✅ PASS" if case.success else "❌ FAIL"
            md.append(f"## {status_icon} - {case.name}\n")
            md.append(
                "<details><summary><b>View Test Case Data</b></summary>\n"
            )

            if case.conversational:
                for turn in case.turns:
                    md.append(f"- **{turn.role.capitalize()}**: {turn.content}")
            else:
                md.append(f"- **Input:** {case.input}")
                md.append(f"- **Actual Output:** {case.actual_output}")

                if case.expected_output and case.expected_output != "N/A":
                    md.append(f"- **Expected Output:** {case.expected_output}")

            md.append("\n</details>\n\n### Metrics\n")
            md.append("| Status | Metric | Score | Threshold | Reason |")
            md.append("|:---:|:---|:---:|:---:|:---|")

            for m in case.metrics_data:
                m_icon = (
                    "✅" if m.success else ("❌" if not m.error else "⚠️ ERROR")
                )
                score_str = f"{m.score:.2f}" if m.score is not None else "N/A"
                thresh_str = (
                    f"{m.threshold:.2f}" if m.threshold is not None else "N/A"
                )
                reason_str = str(m.reason or m.error or "N/A").replace(
                    "\n", " <br> "
                )
                md.append(
                    f"| {m_icon} | **{m.name}** | {score_str} | {thresh_str} | {reason_str} |"
                )

            md.append("\n---\n")

        # Calculate aggregate metrics
        metric_aggregates = {}
        for case in self.test_results:
            for m in case.metrics_data:
                if m.name not in metric_aggregates:
                    metric_aggregates[m.name] = {
                        "total": 0,
                        "passes": 0,
                        "score_sum": 0,
                        "score_count": 0,
                    }

                agg = metric_aggregates[m.name]
                agg["total"] += 1
                if m.success:
                    agg["passes"] += 1
                if m.score is not None:
                    agg["score_sum"] += m.score
                    agg["score_count"] += 1

        if metric_aggregates:
            md.append("## Aggregate Metrics\n")
            md.append("| Metric | Average Score | Pass Rate | Total |")
            md.append("|:---|:---:|:---:|:---:|")

            for metric_name, agg in metric_aggregates.items():
                avg_score = (
                    f"{agg['score_sum'] / agg['score_count']:.2f}"
                    if agg["score_count"] > 0
                    else "N/A"
                )
                pass_rate = (
                    f"{(agg['passes'] / agg['total']) * 100:.2f}%"
                    if agg["total"] > 0
                    else "N/A"
                )
                md.append(
                    f"| **{metric_name}** | {avg_score} | {pass_rate} | {agg['total']} |"
                )

            md.append("\n---\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        print(f"✅ Markdown Dashboard saved to: {filepath}")
