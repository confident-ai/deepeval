import re
from pathlib import Path
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.evaluate.types import TestResult as EvalTestResult
from deepeval.test_run.api import Classification, MetricData


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


def _markdown_table(md: str, heading: str) -> list:
    """Cells of every row of the table under ``heading``, split the way
    GitHub splits them: a ``|`` preceded by a backslash stays in its cell."""
    lines = md.splitlines()
    rows = []
    for line in lines[lines.index(heading) + 1 :]:
        if not line.strip():
            if rows:
                break
            continue
        rows.append(re.split(r"(?<!\\)\|", line.strip())[1:-1])
    return rows


def test_markdown_export_keeps_pipes_and_line_breaks_inside_their_cells(
    tmp_path: Path,
):
    metric = MetricData(
        name="Style | Tone",
        score=0.4,
        threshold=0.5,
        reason="Offers 'a | b' instead of one.\r\nSecond line",
        success=False,
        strictMode=False,
        evaluationModel=None,
        error=None,
        evaluationCost=None,
        verboseLogs=None,
    )
    classification = Classification(
        name="Tone | Safety",
        label="a|b",
        expectedLabel="c|d",
        success=False,
        reason="picked a|b\nnot c|d",
    )
    case = EvalTestResult(
        name="demo",
        success=False,
        input="test input",
        actual_output="test output",
        conversational=False,
        metrics_data=[metric],
        turns=None,
        classifications=[classification],
    )

    EvaluationConsoleReport([case]).export_to_markdown(
        output_dir=str(tmp_path), evaluation_name="pipes"
    )
    md = next(tmp_path.glob("pipes_*.md")).read_text(encoding="utf-8")

    # Header, separator and the one data row must all have the same width.
    for heading, columns in (
        ("### Metrics", 5),
        ("### Classifiers", 5),
        ("## Aggregate Metrics", 4),
        ("## Aggregate Classifiers", 5),
    ):
        rows = _markdown_table(md, heading)
        assert len(rows) == 3, heading
        assert [len(row) for row in rows] == [columns] * 3, heading

    *_, metric_row = _markdown_table(md, "### Metrics")
    assert metric_row[1].strip() == "**Style \\| Tone**"
    assert (
        metric_row[4].strip()
        == "Offers 'a \\| b' instead of one. <br> Second line"
    )

    *_, classifier_row = _markdown_table(md, "### Classifiers")
    assert classifier_row[1].strip() == "**Tone \\| Safety**"
    assert classifier_row[2].strip() == "a\\|b"
    assert classifier_row[3].strip() == "c\\|d"
    assert classifier_row[4].strip() == "picked a\\|b <br> not c\\|d"

    *_, metric_aggregate = _markdown_table(md, "## Aggregate Metrics")
    assert metric_aggregate[2].strip() == "0.00% \\| passed=0 \\| failed=1"
    assert metric_aggregate[3].strip() == "1"

    *_, classifier_aggregate = _markdown_table(md, "## Aggregate Classifiers")
    assert classifier_aggregate[1].strip() == "a\\|b=1"
    assert classifier_aggregate[2].strip() == "0.00% \\| passed=0 \\| failed=1"
    assert classifier_aggregate[3].strip() == "0"
    assert classifier_aggregate[4].strip() == "1"
