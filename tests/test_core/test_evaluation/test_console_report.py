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
    assert list(table.columns[2].cells)[0] == "50.00%"
    assert list(table.columns[3].cells)[0] == "2"
