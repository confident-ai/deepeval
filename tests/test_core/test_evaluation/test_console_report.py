from pathlib import Path
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.evaluate.types import TestResult as EvalTestResult
from deepeval.test_run.api import MetricData


def test_evaluation_console_report_exports(tmp_path: Path):
    metrics_data = [
        MetricData(
            name="Answer Relevancy",
            score=1.0,
            threshold=0.5,
            reason=None,
            success=True,
            strictMode=False,
            evaluationModel=None,
            error=None,
            evaluationCost=None,
            verboseLogs=None,
        )
    ]

    tr = EvalTestResult(
        name="demo",
        success=True,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=metrics_data,
        turns=None,
    )

    console_report = EvaluationConsoleReport([tr])

    # Test HTML export
    console_report.export_to_html(
        output_dir=str(tmp_path), evaluation_name="test_eval"
    )
    html_files = list(tmp_path.glob("test_eval_*.html"))
    assert len(html_files) == 1
    html_content = html_files[0].read_text()
    assert "DeepEval Evaluation Results" in html_content
    assert "demo" in html_content
    assert "Answer Relevancy" in html_content
    assert "Aggregate Metrics" in html_content

    # Test Markdown export
    console_report.export_to_markdown(
        output_dir=str(tmp_path), evaluation_name="test_eval"
    )
    md_files = list(tmp_path.glob("test_eval_*.md"))
    assert len(md_files) == 1
    md_content = md_files[0].read_text()
    assert "DeepEval Evaluation Results" in md_content
    assert "demo" in md_content
    assert "Answer Relevancy" in md_content
    assert "Aggregate Metrics" in md_content


def test_evaluation_console_report_aggregate_metrics():
    metrics_data_1 = [
        MetricData(
            name="Answer Relevancy",
            score=1.0,
            threshold=0.5,
            reason=None,
            success=True,
            strictMode=False,
            evaluationModel=None,
            error=None,
            evaluationCost=None,
            verboseLogs=None,
        )
    ]

    metrics_data_2 = [
        MetricData(
            name="Answer Relevancy",
            score=0.0,
            threshold=0.5,
            reason=None,
            success=False,
            strictMode=False,
            evaluationModel=None,
            error=None,
            evaluationCost=None,
            verboseLogs=None,
        )
    ]

    tr1 = EvalTestResult(
        name="demo1",
        success=True,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=metrics_data_1,
        turns=None,
    )

    tr2 = EvalTestResult(
        name="demo2",
        success=False,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=metrics_data_2,
        turns=None,
    )

    console_report = EvaluationConsoleReport([tr1, tr2])

    # Check if the aggregate table is built correctly
    group = console_report._build_display_elements(truncate=False)

    # The last element should be the aggregate metrics panel
    aggregate_panel = group.renderables[-1]

    # Check if it's a Panel and contains the aggregate metrics table
    assert hasattr(aggregate_panel, "renderable")
    table = aggregate_panel.renderable
    assert "Aggregate Metrics" in str(table.title)

    # The table should have 1 row for "Answer Relevancy"
    # Average score: 0.50, Pass rate: 50.00%, Total: 2
    assert len(table.rows) == 1

    row_data = list(table.columns)
    # columns[0] is Metric, columns[1] is Average Score, columns[2] is Pass Rate, columns[3] is Total
    assert list(table.columns[0].cells)[0] == "Answer Relevancy"
    assert list(table.columns[1].cells)[0] == "0.50"
    assert list(table.columns[2].cells)[0] == "50.00% | passed=1 | failed=1"
    assert list(table.columns[3].cells)[0] == "2"


def _make_metric_named(
    name: str, success: bool, score: float = None
) -> MetricData:
    return MetricData(
        name=name,
        score=score if score is not None else (1.0 if success else 0.0),
        threshold=0.5,
        reason=None,
        success=success,
        strictMode=False,
        evaluationModel=None,
        error=None,
        evaluationCost=None,
        verboseLogs=None,
    )


def _make_metric(success: bool) -> MetricData:
    return MetricData(
        name="Answer Relevancy",
        score=1.0 if success else 0.0,
        threshold=0.5,
        reason=None,
        success=success,
        strictMode=False,
        evaluationModel=None,
        error=None,
        evaluationCost=None,
        verboseLogs=None,
    )


def _make_case(name: str, success: bool) -> EvalTestResult:
    return EvalTestResult(
        name=name,
        success=success,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=[_make_metric(success)],
        turns=None,
    )


def _render_text(report: EvaluationConsoleReport, **kwargs) -> str:
    from rich.console import Console

    console = Console(record=True, width=200)
    console.print(report._build_display_elements(**kwargs))
    return console.export_text()


def test_display_option_failing_hides_passing_cases():
    from deepeval.test_run.test_run import TestRunResultDisplay

    report = EvaluationConsoleReport(
        [_make_case("passing_case", True), _make_case("failing_case", False)]
    )

    text = _render_text(
        report,
        truncate=True,
        display_option=TestRunResultDisplay.FAILING,
    )

    assert "failing_case" in text
    assert "passing_case" not in text
    # Aggregate is computed over ALL cases regardless of the filter.
    assert "Aggregate Metrics" in text


def test_display_option_passing_hides_failing_cases():
    from deepeval.test_run.test_run import TestRunResultDisplay

    report = EvaluationConsoleReport(
        [_make_case("passing_case", True), _make_case("failing_case", False)]
    )

    text = _render_text(
        report,
        truncate=False,
        display_option=TestRunResultDisplay.PASSING,
    )

    assert "passing_case" in text
    assert "failing_case" not in text


def test_display_option_all_shows_every_case():
    from deepeval.test_run.test_run import TestRunResultDisplay

    report = EvaluationConsoleReport(
        [_make_case("passing_case", True), _make_case("failing_case", False)]
    )

    # Explicit ALL and the default should both render every case.
    text_explicit = _render_text(
        report, truncate=False, display_option=TestRunResultDisplay.ALL
    )
    assert "passing_case" in text_explicit
    assert "failing_case" in text_explicit

    text_default = _render_text(report, truncate=False)
    assert "passing_case" in text_default
    assert "failing_case" in text_default


def test_display_option_failing_all_hidden_shows_all_passed_note():
    from deepeval.test_run.test_run import TestRunResultDisplay

    report = EvaluationConsoleReport(
        [_make_case("passing_case_1", True), _make_case("passing_case_2", True)]
    )

    text = _render_text(
        report,
        truncate=True,
        display_option=TestRunResultDisplay.FAILING,
    )

    assert "passing_case" not in text
    assert "All 2 test cases passed — no failing test cases to display." in text


def test_display_option_passing_all_hidden_shows_all_failed_note():
    from deepeval.test_run.test_run import TestRunResultDisplay

    report = EvaluationConsoleReport([_make_case("failing_case", False)])

    text = _render_text(
        report,
        truncate=True,
        display_option=TestRunResultDisplay.PASSING,
    )

    assert "failing_case" not in text
    assert "All 1 test case failed — no passing test cases to display." in text


# ---------------------------------------------------------------------------
# group_by tests
# ---------------------------------------------------------------------------


def _make_tagged_case(
    name: str, success: bool, tags=None, metadata=None
) -> EvalTestResult:
    return EvalTestResult(
        name=name,
        success=success,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=[_make_metric(success)],
        turns=None,
        tags=tags,
        metadata=metadata,
    )


def test_group_by_tag_adds_group_column():
    """group_by a tag adds a 'Group' column and per-tag rows alongside the
    pooled '(all)' row."""
    case_a = _make_tagged_case("case_a", True, tags=["short-answer"])
    case_b = _make_tagged_case("case_b", False, tags=["multi-hop"])
    case_c = _make_tagged_case("case_c", True, tags=["short-answer"])

    report = EvaluationConsoleReport([case_a, case_b, case_c])
    text = _render_text(report, truncate=False, group_by="short-answer")

    # The pooled row is still present
    assert "(all)" in text
    # The group label appears
    assert "short-answer" in text
    # The column header is present
    assert "Group" in text


def test_group_by_tag_correct_counts():
    """Per-group aggregate contains only the cases that carry the tag."""
    case_a = _make_tagged_case("case_a", True, tags=["easy"])
    case_b = _make_tagged_case("case_b", False, tags=["easy"])
    case_c = _make_tagged_case("case_c", True, tags=["hard"])

    report = EvaluationConsoleReport([case_a, case_b, case_c])
    group = report._build_display_elements(truncate=False, group_by="easy")

    agg_panel = group.renderables[-1]
    table = agg_panel.renderable

    # The table has the "(all)" row plus the "easy" row
    assert len(table.rows) == 2

    group_col = list(table.columns[4].cells)
    assert "(all)" in group_col
    assert "easy" in group_col

    # The "easy" row covers 2 cases (case_a and case_b), 1 pass
    total_col = list(table.columns[3].cells)
    pass_rate_col = list(table.columns[2].cells)

    easy_idx = group_col.index("easy")
    assert total_col[easy_idx] == "2"
    assert "passed=1" in pass_rate_col[easy_idx]
    assert "failed=1" in pass_rate_col[easy_idx]


def test_group_by_metadata_key():
    """group_by a metadata key slices by cases whose metadata contains
    that key (any value)."""
    case_a = _make_tagged_case("case_a", True, metadata={"difficulty": "hard"})
    case_b = _make_tagged_case("case_b", False, metadata={"difficulty": "easy"})
    case_c = _make_tagged_case("case_c", True)

    report = EvaluationConsoleReport([case_a, case_b, case_c])
    text = _render_text(report, truncate=False, group_by="difficulty")

    assert "difficulty" in text
    assert "(all)" in text


def test_group_by_no_matching_cases_skips_group_row():
    """A group_by key that matches no cases produces no extra row."""
    case_a = _make_tagged_case("case_a", True, tags=["easy"])

    report = EvaluationConsoleReport([case_a])
    group = report._build_display_elements(
        truncate=False, group_by="nonexistent-tag"
    )
    agg_panel = group.renderables[-1]
    table = agg_panel.renderable

    # Only the pooled "(all)" row; the unmatched group produces nothing
    assert len(table.rows) == 1
    group_col = list(table.columns[4].cells)
    assert group_col[0] == "(all)"


def test_group_by_list_of_tags():
    """group_by accepts a list of tag strings and emits one row per tag."""
    case_a = _make_tagged_case("case_a", True, tags=["easy"])
    case_b = _make_tagged_case("case_b", False, tags=["hard"])
    case_c = _make_tagged_case("case_c", True, tags=["easy", "hard"])

    report = EvaluationConsoleReport([case_a, case_b, case_c])
    text = _render_text(report, truncate=False, group_by=["easy", "hard"])

    assert "easy" in text
    assert "hard" in text
    assert "(all)" in text


def test_group_by_none_preserves_original_table_shape():
    """Without group_by the table has no Group column (backward compat)."""
    case_a = _make_tagged_case("case_a", True, tags=["easy"])

    report = EvaluationConsoleReport([case_a])
    group = report._build_display_elements(truncate=False, group_by=None)
    agg_panel = group.renderables[-1]
    table = agg_panel.renderable

    # Only 4 columns: Metric, Average Score, Pass Rate, Total
    assert len(table.columns) == 4


def test_group_by_markdown_export(tmp_path):
    """export_to_markdown with group_by includes a Group column."""
    case_a = _make_tagged_case("case_a", True, tags=["qa"])
    case_b = _make_tagged_case("case_b", False, tags=["qa"])

    report = EvaluationConsoleReport([case_a, case_b])
    report.export_to_markdown(
        output_dir=str(tmp_path),
        evaluation_name="grouped",
        group_by="qa",
    )
    md_files = list(tmp_path.glob("grouped_*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text()

    assert "Group" in content
    assert "(all)" in content
    assert "qa" in content


def test_tags_propagated_to_test_result():
    """Tags on LLMApiTestCase flow through create_test_result to TestResult."""
    from deepeval.evaluate.utils import create_test_result
    from deepeval.test_run.api import LLMApiTestCase

    api_case = LLMApiTestCase(
        name="tagged",
        input="hi",
        actualOutput="hello",
        success=True,
        metricsData=[],
        tags=["foo", "bar"],
    )
    result = create_test_result(api_case)
    assert result.tags == ["foo", "bar"]
