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
from deepeval.test_run.test_run import TestRunResultDisplay

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


# Shown when a classifier with `allow_none=True` declined to classify.
NO_CLASSIFICATION = "none"


def _classification_label(c) -> str:
    if c.label is not None:
        return str(c.label)
    if c.error:
        return "N/A"
    return NO_CLASSIFICATION


def _classification_status(c, rich: bool = True) -> str:
    if c.error:
        return "[bold red]ERROR[/bold red]" if rich else "⚠️ ERROR"
    if c.success is None:
        # No expected label: the classifier has no pass/fail verdict
        return "[bold dim]NONE[/bold dim]" if rich else "➖ NONE"
    if c.success:
        return "[bold green]PASS[/bold green]" if rich else "✅"
    return "[bold red]FAIL[/bold red]" if rich else "❌"


def _aggregate_classifications(test_results: List[TestResult]) -> dict:
    """Per classifier: label distribution plus pass/fail over rows that carry
    a verdict (an expected label was set)."""
    aggregates: dict = {}
    for case in test_results:
        for c in case.classifications or []:
            agg = aggregates.setdefault(
                c.name,
                {
                    "total": 0,
                    "passes": 0,
                    "fails": 0,
                    "errors": 0,
                    "labels": {},
                },
            )
            agg["total"] += 1
            if c.error:
                agg["errors"] += 1
            else:
                label = _classification_label(c)
                agg["labels"][label] = agg["labels"].get(label, 0) + 1
            if c.success is True:
                agg["passes"] += 1
            elif c.success is False:
                agg["fails"] += 1
    return aggregates


def _format_label_distribution(labels: dict) -> str:
    if not labels:
        return "N/A"
    ordered = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{label}={count}" for label, count in ordered)


def _format_classifier_pass_rate(agg: dict) -> str:
    verdicts = agg["passes"] + agg["fails"]
    if verdicts == 0:
        return "N/A"
    rate = f"{(agg['passes'] / verdicts) * 100:.2f}%"
    return f"{rate} | passed={agg['passes']} | failed={agg['fails']}"


def _format_pass_rate(agg: dict) -> str:
    """Format pass rate with pass/fail counts, each with its flaky sub-count."""
    verdicts = agg["passes"] + agg["fails"]
    rate = f"{(agg['passes'] / verdicts) * 100:.2f}%" if verdicts > 0 else "N/A"
    passed = f"passed={agg['passes']}"
    if agg["flaky_passes"] > 0:
        passed += f" (flaky={agg['flaky_passes']})"
    failed = f"failed={agg['fails']}"
    if agg["flaky_fails"] > 0:
        failed += f" (flaky={agg['flaky_fails']})"
    return f"{rate} | {passed} | {failed}"


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

    @staticmethod
    def _should_skip_case(
        case: TestResult, display_option: TestRunResultDisplay
    ) -> bool:
        """Mirrors ``TestRunManager._should_skip_test_case``."""
        if display_option == TestRunResultDisplay.PASSING and not case.success:
            return True
        elif display_option == TestRunResultDisplay.FAILING and case.success:
            return True
        return False

    def _build_display_elements(
        self,
        truncate: bool = True,
        display_option: TestRunResultDisplay = TestRunResultDisplay.ALL,
    ) -> Group:

        renderables = [
            Panel(
                f"[{DEEPEVAL_PURPLE} bold]🚀 DeepEval Evaluation Results[/{DEEPEVAL_PURPLE} bold]",
                expand=True,
            )
        ]

        displayed_any = False
        for case in self.test_results:
            if self._should_skip_case(case, display_option):
                continue
            displayed_any = True
            metrics_data = case.metrics_data or []
            classifications = case.classifications or []
            if case.success is None:
                # Nothing produced a verdict (e.g. classifiers only, no
                # expected labels): neither passed nor failed.
                status_color = "dim"
                status_icon = "➖"
            else:
                status_color = DEEPEVAL_GREEN if case.success else FAIL_RED
                status_icon = "✅" if case.success else "❌"

            if truncate and case.success:
                summary_parts = []
                if metrics_data:
                    summary_parts.append(f"Passed {len(metrics_data)} metrics")
                if classifications:
                    summary_parts.append(
                        f"{len(classifications)} classifications"
                    )
                summary_text = f"[{status_color} bold]{status_icon} {case.name} ({', '.join(summary_parts)})[/{status_color} bold]"
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

            if metrics_data:
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

                for m in metrics_data:
                    if m.error:
                        m_icon = "[bold red]ERROR[/bold red]"
                    elif m.success is None:
                        # No threshold: the metric has no pass/fail verdict
                        m_icon = "[bold dim]NONE[/bold dim]"
                    elif m.success:
                        m_icon = "[bold green]PASS[/bold green]"
                    else:
                        m_icon = "[bold red]FAIL[/bold red]"

                    score_str = (
                        f"{m.score:.2f}" if m.score is not None else "N/A"
                    )
                    thresh_str = (
                        f"{m.threshold:.2f}"
                        if m.threshold is not None
                        else "N/A"
                    )
                    reason_str = str(m.reason or m.error or "N/A")

                    if truncate and m.success and len(reason_str) > 50:
                        reason_str = reason_str[:47] + "..."

                    metrics_table.add_row(
                        m_icon, m.name, score_str, thresh_str, reason_str
                    )

                content_tree.add(metrics_table)

            if classifications:
                classifiers_table = Table(
                    title="Classifiers",
                    title_justify="left",
                    show_edge=False,
                    header_style=f"bold {DEEPEVAL_PURPLE}",
                    expand=True,
                )
                classifiers_table.add_column("Status", justify="center")
                classifiers_table.add_column("Classifier")
                classifiers_table.add_column("Label")
                classifiers_table.add_column("Expected")
                classifiers_table.add_column("Reason")

                for c in classifications:
                    reason_str = str(c.reason or c.error or "N/A")
                    if truncate and c.success and len(reason_str) > 50:
                        reason_str = reason_str[:47] + "..."
                    classifiers_table.add_row(
                        _classification_status(c),
                        c.name,
                        _classification_label(c),
                        (
                            str(c.expected_label)
                            if c.expected_label is not None
                            else "N/A"
                        ),
                        reason_str,
                    )

                content_tree.add(classifiers_table)

            renderables.append(
                Panel(
                    content_tree,
                    border_style=status_color,
                    padding=(1, 2),
                    expand=True,
                )
            )

        # Only FAILING/PASSING can hide every case.
        if not displayed_any and self.test_results:
            total = len(self.test_results)
            plural = "s" if total != 1 else ""
            if display_option == TestRunResultDisplay.FAILING:
                renderables.append(
                    Panel(
                        f"[{DEEPEVAL_GREEN} bold]✅ All {total} test case{plural} passed — no failing test cases to display.[/{DEEPEVAL_GREEN} bold]",
                        border_style=DEEPEVAL_GREEN,
                        expand=True,
                    )
                )
            elif display_option == TestRunResultDisplay.PASSING:
                renderables.append(
                    Panel(
                        f"[{FAIL_RED} bold]❌ All {total} test case{plural} failed — no passing test cases to display.[/{FAIL_RED} bold]",
                        border_style=FAIL_RED,
                        expand=True,
                    )
                )

        # Calculate aggregate metrics
        metric_aggregates = {}
        for case in self.test_results:
            for m in case.metrics_data or []:
                if m.name not in metric_aggregates:
                    metric_aggregates[m.name] = {
                        "total": 0,
                        "passes": 0,
                        "fails": 0,
                        "flaky_passes": 0,
                        "flaky_fails": 0,
                        "score_sum": 0,
                        "score_count": 0,
                    }

                agg = metric_aggregates[m.name]
                agg["total"] += 1
                # Metrics without a verdict (success=None, e.g. no threshold)
                # don't count towards the pass rate. Flaky metrics count
                # towards their verdict, tracked in a flaky sub-count.
                if m.success is not None:
                    if m.success:
                        agg["passes"] += 1
                        if m.flaky:
                            agg["flaky_passes"] += 1
                    else:
                        agg["fails"] += 1
                        if m.flaky:
                            agg["flaky_fails"] += 1
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
                pass_rate = _format_pass_rate(agg)
                agg_table.add_row(
                    metric_name, avg_score, pass_rate, str(agg["total"])
                )

            renderables.append(
                Panel(agg_table, border_style=DEEPEVAL_PURPLE, expand=True)
            )

        classifier_aggregates = _aggregate_classifications(self.test_results)
        if classifier_aggregates:
            cls_table = Table(
                title="[bold]Aggregate Classifiers[/bold]\n",
                title_justify="left",
                show_edge=False,
                header_style=f"bold {DEEPEVAL_PURPLE}",
                expand=True,
            )
            cls_table.add_column("Classifier")
            cls_table.add_column("Labels")
            cls_table.add_column("Pass Rate")
            cls_table.add_column("Errors")
            cls_table.add_column("Total")

            for name, agg in classifier_aggregates.items():
                cls_table.add_row(
                    name,
                    _format_label_distribution(agg["labels"]),
                    _format_classifier_pass_rate(agg),
                    str(agg["errors"]),
                    str(agg["total"]),
                )

            renderables.append(
                Panel(cls_table, border_style=DEEPEVAL_PURPLE, expand=True)
            )

        return Group(*renderables)

    def render_to_terminal(
        self,
        truncate_passing_cases: bool = True,
        display_option: TestRunResultDisplay = TestRunResultDisplay.ALL,
    ):
        self.console.print()
        self.console.print(
            self._build_display_elements(
                truncate=truncate_passing_cases,
                display_option=display_option,
            )
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
            if case.success is None:
                status_icon = "➖ NONE"
            else:
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

            md.append("\n</details>\n")

            if case.metrics_data:
                md.append("\n### Metrics\n")
                md.append("| Status | Metric | Score | Threshold | Reason |")
                md.append("|:---:|:---|:---:|:---:|:---|")

                for m in case.metrics_data:
                    if m.error:
                        m_icon = "⚠️ ERROR"
                    elif m.success is None:
                        m_icon = "➖ NONE"
                    elif m.success:
                        m_icon = "✅"
                    else:
                        m_icon = "❌"
                    score_str = (
                        f"{m.score:.2f}" if m.score is not None else "N/A"
                    )
                    thresh_str = (
                        f"{m.threshold:.2f}"
                        if m.threshold is not None
                        else "N/A"
                    )
                    reason_str = str(m.reason or m.error or "N/A").replace(
                        "\n", " <br> "
                    )
                    md.append(
                        f"| {m_icon} | **{m.name}** | {score_str} | {thresh_str} | {reason_str} |"
                    )

            if case.classifications:
                md.append("\n### Classifiers\n")
                md.append("| Status | Classifier | Label | Expected | Reason |")
                md.append("|:---:|:---|:---:|:---:|:---|")
                for c in case.classifications:
                    reason_str = str(c.reason or c.error or "N/A").replace(
                        "\n", " <br> "
                    )
                    label_str = _classification_label(c)
                    expected_str = (
                        str(c.expected_label)
                        if c.expected_label is not None
                        else "N/A"
                    )
                    md.append(
                        f"| {_classification_status(c, rich=False)} | **{c.name}** | {label_str} | {expected_str} | {reason_str} |"
                    )

            md.append("\n---\n")

        # Calculate aggregate metrics
        metric_aggregates = {}
        for case in self.test_results:
            for m in case.metrics_data or []:
                if m.name not in metric_aggregates:
                    metric_aggregates[m.name] = {
                        "total": 0,
                        "passes": 0,
                        "fails": 0,
                        "flaky_passes": 0,
                        "flaky_fails": 0,
                        "score_sum": 0,
                        "score_count": 0,
                    }

                agg = metric_aggregates[m.name]
                agg["total"] += 1
                # Metrics without a verdict (success=None, e.g. no threshold)
                # don't count towards the pass rate. Flaky metrics count
                # towards their verdict, tracked in a flaky sub-count.
                if m.success is not None:
                    if m.success:
                        agg["passes"] += 1
                        if m.flaky:
                            agg["flaky_passes"] += 1
                    else:
                        agg["fails"] += 1
                        if m.flaky:
                            agg["flaky_fails"] += 1
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
                pass_rate = _format_pass_rate(agg)
                md.append(
                    f"| **{metric_name}** | {avg_score} | {pass_rate} | {agg['total']} |"
                )

            md.append("\n---\n")

        classifier_aggregates = _aggregate_classifications(self.test_results)
        if classifier_aggregates:
            md.append("## Aggregate Classifiers\n")
            md.append("| Classifier | Labels | Pass Rate | Errors | Total |")
            md.append("|:---|:---|:---:|:---:|:---:|")
            for name, agg in classifier_aggregates.items():
                md.append(
                    f"| **{name}** | {_format_label_distribution(agg['labels'])} | {_format_classifier_pass_rate(agg)} | {agg['errors']} | {agg['total']} |"
                )
            md.append("\n---\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        print(f"✅ Markdown Dashboard saved to: {filepath}")
