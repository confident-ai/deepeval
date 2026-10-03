from __future__ import annotations

import json
import os
import re
import sys
import time
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.terminal_theme import TerminalTheme

from deepeval.test_run.test_run import TestRun, TestRunEncoder
from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase, MetricData
from deepeval.evaluate.types import MetricDiff, LLMTestCaseDiff, RunComparisonResult
from deepeval.evaluate.console_report import LIGHT_THEME, DEEPEVAL_PURPLE, DEEPEVAL_GREEN, FAIL_RED

RUN_FILE_REGEX = re.compile(r"^test_run_\d{8}_\d{6}(?:_\d+)?\.json$")

def find_latest_runs(folder: Path) -> Tuple[Path, Path]:
    """Resolves and returns paths to the base and latest test run JSON files.
    
    Optimized to match runs belonging to the same test suite (same test_file).
    Scans backwards up to 50 files and skips malformed JSON files gracefully.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder '{folder}' does not exist or is not a directory.")
        
    files = [f for f in folder.glob("*.json") if RUN_FILE_REGEX.match(f.name)]
    if not files:
        raise FileNotFoundError(f"No test_run_*.json files found in folder: {folder}")
    
    # Sort chronologically
    files.sort(key=lambda x: x.name)
    if len(files) < 2:
        raise ValueError(
            f"Found only {len(files)} test run file(s) in {folder}. "
            "Comparison requires at least two runs to execute."
        )
        
    latest_file = files[-1]
    
    # Attempt to load the latest run's test file metadata
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            latest_data = json.load(f)
        latest_test_file = latest_data.get("testFile") or latest_data.get("test_file")
    except Exception:
        # Fallback to simple second-latest if latest is corrupted
        return files[-2], latest_file

    # Scan backwards to find the most recent previous run of the same test file
    scanned = 0
    for prev_file in reversed(files[:-1]):
        scanned += 1
        if scanned > 50:
            break
        try:
            with open(prev_file, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_test_file = prev_data.get("testFile") or prev_data.get("test_file")
            if prev_test_file == latest_test_file:
                return prev_file, latest_file
        except Exception:
            # Skip malformed/corrupted files during chronological scan
            continue

    # Fallback to the second-latest run if no match found
    return files[-2], latest_file

def _get_case_signature(case: Union[LLMApiTestCase, ConversationalApiTestCase]) -> str:
    if isinstance(case, LLMApiTestCase):
        return case.name or case.input or f"order_{case.order}"
    else:
        if case.name:
            return case.name
        sig_str = f"scenario:{case.scenario or ''}_turns:"
        for turn in case.turns:
            sig_str += f"{turn.role}:{turn.content or ''}|"
        return sig_str

def _build_cases_map(cases: List[Union[LLMApiTestCase, ConversationalApiTestCase]]) -> Dict[str, Union[LLMApiTestCase, ConversationalApiTestCase]]:
    counts = {}
    cases_map = {}
    for case in cases:
        sig = _get_case_signature(case)
        if sig in counts:
            counts[sig] += 1
            unique_sig = f"{sig}_collision_{counts[sig]}"
        else:
            counts[sig] = 1
            unique_sig = sig
        cases_map[unique_sig] = case
    return cases_map

def compare_runs(path_a: Union[str, Path], path_b: Union[str, Path]) -> RunComparisonResult:
    """Parses and compares two test runs, generating a RunComparisonResult.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    
    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)
        
    try:
        run_a = TestRun.model_validate(data_a)
    except AttributeError:
        run_a = TestRun.parse_obj(data_a)
        
    try:
        run_b = TestRun.model_validate(data_b)
    except AttributeError:
        run_b = TestRun.parse_obj(data_b)

    # Initialize comparison result
    comparison = RunComparisonResult(
        run_a_name=run_a.identifier or path_a.stem,
        run_b_name=run_b.identifier or path_b.stem,
        old_passed=run_a.test_passed or 0,
        new_passed=run_b.test_passed or 0,
        old_failed=run_a.test_failed or 0,
        new_failed=run_b.test_failed or 0,
        old_duration=run_a.run_duration or 0.0,
        new_duration=run_b.run_duration or 0.0,
        old_cost=run_a.evaluation_cost,
        new_cost=run_b.evaluation_cost,
    )

    # Map test cases
    all_cases_a = run_a.test_cases + run_a.conversational_test_cases
    all_cases_b = run_b.test_cases + run_b.conversational_test_cases
    
    map_a = _build_cases_map(all_cases_a)
    map_b = _build_cases_map(all_cases_b)
    
    all_sigs = set(map_a.keys()) | set(map_b.keys())
    
    # Track metrics for average calculations
    metric_scores_a: Dict[str, List[float]] = {}
    metric_scores_b: Dict[str, List[float]] = {}

    for sig in all_sigs:
        case_a = map_a.get(sig)
        case_b = map_b.get(sig)
        
        diff = LLMTestCaseDiff(name=sig)
        
        if case_a is not None and case_b is not None:
            # Common test case in both runs
            diff.is_conversational = isinstance(case_b, ConversationalApiTestCase)
            diff.input = case_b.input if isinstance(case_b, LLMApiTestCase) else f"Conversational Scenario: {case_b.scenario or 'N/A'}"
            diff.old_success = case_a.success
            diff.new_success = case_b.success
            diff.old_cost = case_a.evaluation_cost
            diff.new_cost = case_b.evaluation_cost
            diff.old_latency = case_a.run_duration
            diff.new_latency = case_b.run_duration
            
            metrics_a = {m.name: m for m in case_a.metrics_data} if case_a.metrics_data else {}
            metrics_b = {m.name: m for m in case_b.metrics_data} if case_b.metrics_data else {}
            all_metric_names = set(metrics_a.keys()) | set(metrics_b.keys())
            
            has_regression = False
            has_improvement = False
            
            for m_name in all_metric_names:
                m_a = metrics_a.get(m_name)
                m_b = metrics_b.get(m_name)
                
                m_diff = MetricDiff(name=m_name)
                if m_a is not None:
                    m_diff.old_score = m_a.score
                    m_diff.old_success = m_a.success
                    m_diff.old_reason = m_a.reason
                    m_diff.old_error = m_a.error
                    if m_a.score is not None:
                        metric_scores_a.setdefault(m_name, []).append(m_a.score)
                        
                if m_b is not None:
                    m_diff.new_score = m_b.score
                    m_diff.new_success = m_b.success
                    m_diff.new_reason = m_b.reason
                    m_diff.new_error = m_b.error
                    if m_b.score is not None:
                        metric_scores_b.setdefault(m_name, []).append(m_b.score)
                
                if m_a is not None and m_b is not None:
                    if m_a.score is not None and m_b.score is not None:
                        m_diff.diff = round(m_b.score - m_a.score, 4)
                        if m_diff.diff < -0.0001:
                            has_regression = True
                        elif m_diff.diff > 0.0001:
                            has_improvement = True
                    
                    if m_a.success is True and m_b.success is False:
                        has_regression = True
                    elif m_a.success is False and m_b.success is True:
                        has_improvement = True
                elif m_a is not None:
                    # Metric was removed in the new run
                    pass
                elif m_b is not None:
                    # Metric was added in the new run
                    has_improvement = True

                diff.metrics[m_name] = m_diff
                
            if diff.old_success is True and diff.new_success is False:
                diff.change_status = "degraded"
            elif diff.old_success is False and diff.new_success is True:
                diff.change_status = "improved"
            elif has_regression:
                diff.change_status = "degraded"
            elif has_improvement:
                diff.change_status = "improved"
            else:
                diff.change_status = "unchanged"
                
        elif case_a is None and case_b is not None:
            # Added test case
            diff.is_conversational = isinstance(case_b, ConversationalApiTestCase)
            diff.input = case_b.input if isinstance(case_b, LLMApiTestCase) else f"Conversational Scenario: {case_b.scenario or 'N/A'}"
            diff.new_success = case_b.success
            diff.new_cost = case_b.evaluation_cost
            diff.new_latency = case_b.run_duration
            diff.change_status = "added"
            
            for m in (case_b.metrics_data or []):
                diff.metrics[m.name] = MetricDiff(
                    name=m.name,
                    new_score=m.score,
                    new_success=m.success,
                    new_reason=m.reason,
                    new_error=m.error
                )
                if m.score is not None:
                    metric_scores_b.setdefault(m.name, []).append(m.score)
                    
        elif case_a is not None and case_b is None:
            # Removed test case
            diff.is_conversational = isinstance(case_a, ConversationalApiTestCase)
            diff.input = case_a.input if isinstance(case_a, LLMApiTestCase) else f"Conversational Scenario: {case_a.scenario or 'N/A'}"
            diff.old_success = case_a.success
            diff.old_cost = case_a.evaluation_cost
            diff.old_latency = case_a.run_duration
            diff.change_status = "removed"
            
            for m in (case_a.metrics_data or []):
                diff.metrics[m.name] = MetricDiff(
                    name=m.name,
                    old_score=m.score,
                    old_success=m.success,
                    old_reason=m.reason,
                    old_error=m.error
                )
                if m.score is not None:
                    metric_scores_a.setdefault(m.name, []).append(m.score)
                    
        comparison.case_diffs.append(diff)
        
    # Calculate metric averages summaries
    all_metric_names = set(metric_scores_a.keys()) | set(metric_scores_b.keys())
    for m_name in all_metric_names:
        avg_a = round(sum(metric_scores_a[m_name]) / len(metric_scores_a[m_name]), 4) if m_name in metric_scores_a and metric_scores_a[m_name] else 0.0
        avg_b = round(sum(metric_scores_b[m_name]) / len(metric_scores_b[m_name]), 4) if m_name in metric_scores_b and metric_scores_b[m_name] else 0.0
        comparison.metric_summaries[m_name] = (avg_a, avg_b)
        
    return comparison

class ComparativeConsoleReport:
    def __init__(self, comparison: RunComparisonResult):
        self.comparison = comparison
        self.console = Console()

    def _build_display_elements(self, status_filter: str = "all") -> Group:
        renderables = [
            Panel(
                f"[{DEEPEVAL_PURPLE} bold]⚖️ DeepEval Local Run Comparison[/{DEEPEVAL_PURPLE} bold]\n"
                f"[dim]Base Run (A): {self.comparison.run_a_name}[/dim]\n"
                f"[dim]New Run  (B): {self.comparison.run_b_name}[/dim]",
                expand=True,
            )
        ]

        # 1. Summary comparison Table
        summary_table = Table(
            title="[bold]Run Summary Comparison[/bold]",
            title_justify="left",
            show_edge=False,
            header_style=f"bold {DEEPEVAL_PURPLE}",
            expand=True,
        )
        summary_table.add_column("Metric Category")
        summary_table.add_column("Base (A)")
        summary_table.add_column("New (B)")
        summary_table.add_column("Delta")

        # Passes diff
        passed_diff = self.comparison.new_passed - self.comparison.old_passed
        passed_diff_str = f"+{passed_diff}" if passed_diff > 0 else str(passed_diff)
        passed_color = "green" if passed_diff >= 0 else "red"
        
        # Fails diff
        failed_diff = self.comparison.new_failed - self.comparison.old_failed
        failed_diff_str = f"+{failed_diff}" if failed_diff > 0 else str(failed_diff)
        failed_color = "green" if failed_diff <= 0 else "red"

        # Duration diff
        duration_diff = self.comparison.new_duration - self.comparison.old_duration
        duration_diff_str = f"{duration_diff:+.2f}s"
        duration_color = "green" if duration_diff <= 0 else "red"

        # Cost diff
        cost_a = self.comparison.old_cost
        cost_b = self.comparison.new_cost
        if cost_a is not None and cost_b is not None:
            cost_diff = cost_b - cost_a
            cost_diff_str = f"${cost_diff:+.4f}"
            cost_color = "green" if cost_diff <= 0.0001 else "red"
            cost_a_str = f"${cost_a:.4f}"
            cost_b_str = f"${cost_b:.4f}"
        else:
            cost_diff_str = "N/A"
            cost_color = "dim"
            cost_a_str = f"${cost_a:.4f}" if cost_a is not None else "N/A"
            cost_b_str = f"${cost_b:.4f}" if cost_b is not None else "N/A"

        summary_table.add_row("Passed Tests", str(self.comparison.old_passed), str(self.comparison.new_passed), f"[{passed_color}]{passed_diff_str}[/{passed_color}]")
        summary_table.add_row("Failed Tests", str(self.comparison.old_failed), str(self.comparison.new_failed), f"[{failed_color}]{failed_diff_str}[/{failed_color}]")
        summary_table.add_row("Duration", f"{self.comparison.old_duration:.2f}s", f"{self.comparison.new_duration:.2f}s", f"[{duration_color}]{duration_diff_str}[/{duration_color}]")
        summary_table.add_row("Evaluation Cost", cost_a_str, cost_b_str, f"[{cost_color}]{cost_diff_str}[/{cost_color}]")

        renderables.append(Panel(summary_table, border_style=DEEPEVAL_PURPLE, expand=True))

        # 2. Average Metrics Table
        if self.comparison.metric_summaries:
            metrics_table = Table(
                title="[bold]Metric Averages Comparison[/bold]",
                title_justify="left",
                show_edge=False,
                header_style=f"bold {DEEPEVAL_PURPLE}",
                expand=True,
            )
            metrics_table.add_column("Metric")
            metrics_table.add_column("Base Avg (A)")
            metrics_table.add_column("New Avg (B)")
            metrics_table.add_column("Delta")

            for m_name, (avg_a, avg_b) in self.comparison.metric_summaries.items():
                m_diff = round(avg_b - avg_a, 4)
                m_diff_str = f"{m_diff:+.3f}"
                if m_diff > 0.0001:
                    m_color = "green"
                elif m_diff < -0.0001:
                    m_color = "red"
                else:
                    m_color = "white"
                    m_diff_str = "0.000"

                metrics_table.add_row(m_name, f"{avg_a:.3f}", f"{avg_b:.3f}", f"[{m_color}]{m_diff_str}[/{m_color}]")

            renderables.append(Panel(metrics_table, border_style=DEEPEVAL_PURPLE, expand=True))

        # 3. Detailed test case diffs
        # Group diffs by category
        regressed = [d for d in self.comparison.case_diffs if d.change_status == "degraded"]
        improved = [d for d in self.comparison.case_diffs if d.change_status == "improved"]
        added = [d for d in self.comparison.case_diffs if d.change_status == "added"]
        removed = [d for d in self.comparison.case_diffs if d.change_status == "removed"]
        unchanged = [d for d in self.comparison.case_diffs if d.change_status == "unchanged"]

        # Tree layout for detailed comparison
        detail_tree = Tree("[bold underline]Detailed Test Case Changes[/bold underline]")

        def _add_cases_to_tree(cases: List[LLMTestCaseDiff], tree_node_title: str):
            node = detail_tree.add(tree_node_title)
            for diff in cases:
                case_node_title = f"[bold]{diff.name}[/bold]"
                if diff.change_status == "added":
                    case_node_title += " [green](Added)[/green]"
                elif diff.change_status == "removed":
                    case_node_title += " [red](Removed)[/red]"
                elif diff.change_status == "degraded":
                    case_node_title += " [red](Degraded)[/red]"
                elif diff.change_status == "improved":
                    case_node_title += " [green](Improved)[/green]"
                
                case_node = node.add(case_node_title)
                case_node.add(f"[cyan]Input / Turns Preview:[/cyan] {diff.input or 'N/A'}")
                
                t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
                t.add_column("Metric")
                t.add_column("Base Score")
                t.add_column("New Score")
                t.add_column("Delta")
                t.add_column("Status Change")

                for m_name, m_diff in diff.metrics.items():
                    score_a = f"{m_diff.old_score:.3f}" if m_diff.old_score is not None else "N/A"
                    score_b = f"{m_diff.new_score:.3f}" if m_diff.new_score is not None else "N/A"
                    
                    if m_diff.diff is not None:
                        d_val = m_diff.diff
                        d_str = f"{d_val:+.3f}"
                        d_color = "green" if d_val > 0.0001 else ("red" if d_val < -0.0001 else "white")
                    else:
                        d_str = "N/A"
                        d_color = "dim"

                    status_a = "PASS" if m_diff.old_success else ("FAIL" if m_diff.old_success is False else "N/A")
                    status_b = "PASS" if m_diff.new_success else ("FAIL" if m_diff.new_success is False else "N/A")
                    
                    status_change = f"{status_a} -> {status_b}"
                    if m_diff.old_success is True and m_diff.new_success is False:
                        status_change = f"[red]{status_change} (Degraded)[/red]"
                    elif m_diff.old_success is False and m_diff.new_success is True:
                        status_change = f"[green]{status_change} (Improved)[/green]"

                    t.add_row(m_name, score_a, score_b, f"[{d_color}]{d_str}[/{d_color}]", status_change)
                
                case_node.add(t)

        # Filtering logic
        if status_filter == "all" or status_filter == "regressed":
            if regressed:
                _add_cases_to_tree(regressed, f"⚠️ [bold red]Regressions ({len(regressed)})[/bold red]")
        if status_filter == "all" or status_filter == "improved":
            if improved:
                _add_cases_to_tree(improved, f"📈 [bold green]Improvements ({len(improved)})[/bold green]")
        if status_filter == "all":
            if added:
                _add_cases_to_tree(added, f"➕ [bold yellow]Added Test Cases ({len(added)})[/bold yellow]")
            if removed:
                _add_cases_to_tree(removed, f"➖ [bold dim]Removed Test Cases ({len(removed)})[/bold dim]")
            if unchanged:
                _add_cases_to_tree(unchanged, f"🔒 [bold white]Unchanged Test Cases ({len(unchanged)})[/bold white]")

        renderables.append(Panel(detail_tree, border_style=DEEPEVAL_PURPLE, expand=True))
        return Group(*renderables)

    def render_to_terminal(self, status_filter: str = "all"):
        self.console.print()
        self.console.print(self._build_display_elements(status_filter=status_filter))
        self.console.print()

    def export_to_html(self, output_dir: str, evaluation_name: str = "comparison"):
        os.makedirs(output_dir, exist_ok=True)
        safe_name = str(evaluation_name).replace(" ", "_").lower()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{safe_name}_{timestamp}.html")

        dummy_file = io.StringIO()
        html_console = Console(record=True, file=dummy_file, force_terminal=True)
        html_console.print(self._build_display_elements())
        html_console.save_html(filepath, theme=LIGHT_THEME)

        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        css_patch = "<style>pre { line-height: 1.1 !important; }</style></head>"
        html_content = html_content.replace("</head>", css_patch)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Comparative HTML dashboard saved to: {filepath}")

    def export_to_json(self, output_dir_or_file: str, evaluation_name: str = "comparison"):
        data = self.comparison.model_dump(by_alias=True)
        
        if output_dir_or_file == "-":
            # Stream directly to stdout
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
            return
            
        path = Path(output_dir_or_file)
        if path.is_dir() or not path.suffix:
            # It's a directory target
            os.makedirs(path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = str(evaluation_name).replace(" ", "_").lower()
            filepath = path / f"{safe_name}_{timestamp}.json"
        else:
            # It's a specific file target
            os.makedirs(path.parent, exist_ok=True)
            filepath = path

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Comparative JSON report saved to: {filepath}")

    def export_to_markdown(self, output_path_or_dir: str, evaluation_name: str = "comparison"):
        passed_diff = self.comparison.new_passed - self.comparison.old_passed
        failed_diff = self.comparison.new_failed - self.comparison.old_failed
        dur_diff = self.comparison.new_duration - self.comparison.old_duration

        cost_a = self.comparison.old_cost
        cost_b = self.comparison.new_cost
        cost_diff_str = f"${cost_b - cost_a:+.4f}" if cost_a is not None and cost_b is not None else "N/A"
        cost_a_str = f"${cost_a:.4f}" if cost_a is not None else "N/A"
        cost_b_str = f"${cost_b:.4f}" if cost_b is not None else "N/A"

        md = [
            f"# ⚖️ DeepEval Comparative Report: {evaluation_name}\n",
            f"- **Base Run (A):** `{self.comparison.run_a_name}`",
            f"- **New Run  (B):** `{self.comparison.run_b_name}`\n",
            "## Summary statistics Comparison\n",
            "| Category | Base (A) | New (B) | Delta |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Passed Tests** | {self.comparison.old_passed} | {self.comparison.new_passed} | {passed_diff:+} |",
            f"| **Failed Tests** | {self.comparison.old_failed} | {self.comparison.new_failed} | {failed_diff:+} |",
            f"| **Duration** | {self.comparison.old_duration:.2f}s | {self.comparison.new_duration:.2f}s | {dur_diff:+.2f}s |",
            f"| **Evaluation Cost** | {cost_a_str} | {cost_b_str} | {cost_diff_str} |"
        ]

        if self.comparison.metric_summaries:
            md.append("\n## Metric Averages Comparison\n")
            md.append("| Metric | Base Avg (A) | New Avg (B) | Delta |")
            md.append("| :--- | :--- | :--- | :--- |")
            for m_name, (avg_a, avg_b) in self.comparison.metric_summaries.items():
                m_diff = round(avg_b - avg_a, 4)
                md.append(f"| **{m_name}** | {avg_a:.3f} | {avg_b:.3f} | {m_diff:+.3f} |")

        md.append("\n## Detailed Test Case Changes\n")
        
        for diff in self.comparison.case_diffs:
            status_map = {
                "added": "➕ Added",
                "removed": "➖ Removed",
                "degraded": "⚠️ Degraded",
                "improved": "📈 Improved",
                "unchanged": "🔒 Unchanged"
            }
            icon = status_map.get(diff.change_status, "🔒")
            md.append(f"### {icon} - {diff.name}\n")
            md.append(f"- **Input / Turns Preview:** {diff.input or 'N/A'}\n")
            
            md.append("| Metric | Base Score | New Score | Delta | Status Change |")
            md.append("| :--- | :---: | :---: | :---: | :--- |")
            for m_name, m_diff in diff.metrics.items():
                score_a = f"{m_diff.old_score:.3f}" if m_diff.old_score is not None else "N/A"
                score_b = f"{m_diff.new_score:.3f}" if m_diff.new_score is not None else "N/A"
                d_str = f"{m_diff.diff:+.3f}" if m_diff.diff is not None else "N/A"
                
                status_a = "PASS" if m_diff.old_success else ("FAIL" if m_diff.old_success is False else "N/A")
                status_b = "PASS" if m_diff.new_success else ("FAIL" if m_diff.new_success is False else "N/A")
                
                md.append(f"| {m_name} | {score_a} | {score_b} | {d_str} | {status_a} &rarr; {status_b} |")
            
            md.append("\n---\n")

        markdown_body = "\n".join(md) + "\n"

        if output_path_or_dir == "-":
            # Stream directly to stdout
            sys.stdout.write(markdown_body)
            return

        path = Path(output_path_or_dir)
        if path.is_dir() or not path.suffix:
            # It's a directory target
            os.makedirs(path, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = str(evaluation_name).replace(" ", "_").lower()
            filepath = path / f"{safe_name}_{timestamp}.md"
        else:
            # It's a specific file path
            os.makedirs(path.parent, exist_ok=True)
            filepath = path

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_body)
        print(f"✅ Comparative Markdown saved to: {filepath}")
