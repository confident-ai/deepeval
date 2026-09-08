import os
import json
import pytest
from pathlib import Path
from deepeval.evaluate.compare_runs import (
    compare_runs,
    find_latest_runs,
    ComparativeConsoleReport,
    RunComparisonResult,
)
from deepeval.test_run.test_run import TestRun
from deepeval.test_run.api import LLMApiTestCase, ConversationalApiTestCase, MetricData

@pytest.fixture
def temp_run_dir(tmp_path):
    """Creates a temporary directory for test runs."""
    return tmp_path

def test_find_latest_runs_strict_regex(temp_run_dir):
    # Test regex filters out non-timestamped json files
    file1 = temp_run_dir / "test_run_20260721_120000.json"
    file1.write_text("{}", encoding="utf-8")
    
    file2 = temp_run_dir / "test_run_20260721_120100.json"
    file2.write_text("{}", encoding="utf-8")
    
    # Random json file to ignore
    file_ignored = temp_run_dir / "config.json"
    file_ignored.write_text("{}", encoding="utf-8")
    
    resolved_a, resolved_b = find_latest_runs(temp_run_dir)
    assert resolved_a == file1
    assert resolved_b == file2

    # Check that sorting works properly chronologically
    file3 = temp_run_dir / "test_run_20260721_120200.json"
    file3.write_text("{}", encoding="utf-8")
    
    resolved_a, resolved_b = find_latest_runs(temp_run_dir)
    assert resolved_a == file2
    assert resolved_b == file3

def test_compare_runs_logic(temp_run_dir):
    # Create mock Run A (Base)
    case_a1 = LLMApiTestCase(
        name="Case 1",
        input="Hello",
        actualOutput="Hi",
        success=True,
        metricsData=[
            MetricData(name="Faithfulness", score=0.9, threshold=0.7, success=True, reason="Good faithfulness")
        ],
        runDuration=1.5,
        evaluationCost=0.002
    )
    case_a2 = LLMApiTestCase(
        name="Case 2",
        input="Query",
        actualOutput="Answer",
        success=False,
        metricsData=[
            MetricData(name="Answer Relevancy", score=0.5, threshold=0.7, success=False, reason="Irrelevant answer")
        ],
        runDuration=2.0,
        evaluationCost=0.003
    )
    
    run_a = TestRun(
        testFile="test_sample.py",
        testCases=[case_a1, case_a2],
        testPassed=1,
        testFailed=1,
        runDuration=3.5,
        evaluationCost=0.005,
        identifier="Run A Base"
    )

    # Create mock Run B (New) - case 1 degraded, case 2 improved, case 3 added
    case_b1 = LLMApiTestCase(
        name="Case 1",
        input="Hello",
        actualOutput="Hi",
        success=False,
        metricsData=[
            MetricData(name="Faithfulness", score=0.6, threshold=0.7, success=False, reason="Low faithfulness")
        ],
        runDuration=1.2,
        evaluationCost=0.001
    )
    case_b2 = LLMApiTestCase(
        name="Case 2",
        input="Query",
        actualOutput="Answer",
        success=True,
        metricsData=[
            MetricData(name="Answer Relevancy", score=0.85, threshold=0.7, success=True, reason="Very relevant answer")
        ],
        runDuration=1.8,
        evaluationCost=0.002
    )
    case_b3 = LLMApiTestCase(
        name="Case 3",
        input="New case",
        actualOutput="New output",
        success=True,
        metricsData=[
            MetricData(name="Faithfulness", score=0.95, threshold=0.7, success=True, reason="Excellent")
        ],
        runDuration=0.8,
        evaluationCost=0.001
    )

    run_b = TestRun(
        testFile="test_sample.py",
        testCases=[case_b1, case_b2, case_b3],
        testPassed=2,
        testFailed=1,
        runDuration=3.8,
        evaluationCost=0.004,
        identifier="Run B New"
    )

    # Dump runs to temporary files
    path_a = temp_run_dir / "run_a.json"
    path_b = temp_run_dir / "run_b.json"
    
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(run_a.model_dump(by_alias=True, exclude_none=True), f)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(run_b.model_dump(by_alias=True, exclude_none=True), f)

    # Run comparison
    comparison = compare_runs(path_a, path_b)

    assert comparison.run_a_name == "Run A Base"
    assert comparison.run_b_name == "Run B New"
    assert comparison.old_passed == 1
    assert comparison.new_passed == 2
    assert comparison.old_failed == 1
    assert comparison.new_failed == 1
    assert comparison.old_duration == 3.5
    assert comparison.new_duration == 3.8
    assert comparison.old_cost == 0.005
    assert comparison.new_cost == 0.004

    assert "Faithfulness" in comparison.metric_summaries
    assert "Answer Relevancy" in comparison.metric_summaries
    
    avg_faith_a, avg_faith_b = comparison.metric_summaries["Faithfulness"]
    assert avg_faith_a == pytest.approx(0.9)
    assert avg_faith_b == pytest.approx(0.775)

    # Check case diffs
    diffs = {d.name: d for d in comparison.case_diffs}
    assert len(diffs) == 3
    assert "Case 1" in diffs
    assert "Case 2" in diffs
    assert "Case 3" in diffs

    # Case 1 (Degraded)
    c1 = diffs["Case 1"]
    assert c1.change_status == "degraded"
    assert c1.old_success is True
    assert c1.new_success is False
    assert "Faithfulness" in c1.metrics
    assert c1.metrics["Faithfulness"].diff == pytest.approx(-0.3)

    # Case 2 (Improved)
    c2 = diffs["Case 2"]
    assert c2.change_status == "improved"
    assert c2.old_success is False
    assert c2.new_success is True
    assert "Answer Relevancy" in c2.metrics
    assert c2.metrics["Answer Relevancy"].diff == pytest.approx(0.35)

    # Case 3 (Added)
    c3 = diffs["Case 3"]
    assert c3.change_status == "added"
    assert c3.old_success is None
    assert c3.new_success is True

    # 3. Test Console report and Exports
    report = ComparativeConsoleReport(comparison)
    
    # Render with status filters should execute without error
    report.render_to_terminal(status_filter="all")
    report.render_to_terminal(status_filter="regressed")
    
    # Export html
    report.export_to_html(str(temp_run_dir), evaluation_name="test_comparison")
    html_files = list(temp_run_dir.glob("test_comparison_*.html"))
    assert len(html_files) == 1
    
    # Export markdown
    report.export_to_markdown(str(temp_run_dir), evaluation_name="test_comparison")
    md_files = list(temp_run_dir.glob("test_comparison_*.md"))
    assert len(md_files) == 1
    
    md_content = md_files[0].read_text(encoding="utf-8")
    assert "Base Run (A):" in md_content
    assert "New Run  (B):" in md_content
    assert "Summary statistics Comparison" in md_content

    # Export JSON
    report.export_to_json(str(temp_run_dir), evaluation_name="test_comparison")
    json_files = list(temp_run_dir.glob("test_comparison_*.json"))
    assert len(json_files) == 1
    
    json_data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert json_data["run_a_name"] == "Run A Base"
    assert json_data["run_b_name"] == "Run B New"
    assert len(json_data["case_diffs"]) == 3

    # Test stdout streaming (should execute without file errors)
    report.export_to_markdown("-")
    report.export_to_json("-")

def test_compare_runs_empty_edge_cases(temp_run_dir):
    # Empty runs comparison should execute gracefully
    run_a = TestRun(testFile="test_empty.py", testCases=[], testPassed=0, testFailed=0)
    run_b = TestRun(testFile="test_empty.py", testCases=[], testPassed=0, testFailed=0)

    path_a = temp_run_dir / "empty_a.json"
    path_b = temp_run_dir / "empty_b.json"
    
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(run_a.model_dump(by_alias=True, exclude_none=True), f)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(run_b.model_dump(by_alias=True, exclude_none=True), f)

    comparison = compare_runs(path_a, path_b)
    assert len(comparison.case_diffs) == 0
    assert len(comparison.metric_summaries) == 0
