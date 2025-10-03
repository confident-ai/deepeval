import re
from pathlib import Path
from deepeval.evaluate.utils import print_test_result, write_test_result_to_file
from deepeval.evaluate.types import TestResult as EvalTestResult
from deepeval.test_run.api import TurnApi
from deepeval.test_run.test_run import TestRunResultDisplay as RunResultDisplay
from deepeval.test_case import ToolCall


def T(order, role, content, tools=None):
    return TurnApi(
        order=order,
        role=role,
        content=content,
        toolsCalled=tools,  # <- validation only happens on alias
    )


def test_print_test_result_conversational_turns_are_sorted_and_prefixed(capsys):
    turns = [
        T(2, "assistant", "C", [ToolCall(name="a"), ToolCall(name="b")]),
        T(0, "user", "A"),
        T(1, "assistant", "B"),
    ]

    # sanity check the data before asserting on printed output
    assert turns[0].order == 2
    assert turns[0].tools_called and [
        tc.name for tc in turns[0].tools_called
    ] == ["a", "b"]

    tr = EvalTestResult(
        name="demo",
        success=True,
        input=None,
        conversational=True,
        metrics_data=[],
        turns=turns,
    )
    print_test_result(tr, display=RunResultDisplay.ALL)
    out = capsys.readouterr().out

    assert "For conversational test case:" in out
    assert "  Turns:" in out
    # we only expect tool printing on the turn that had tools
    # it’s the order=2 line
    assert "  | tools: a, b" in out
    assert re.search(r"\n\s*0\.", out)
    assert re.search(r"\n\s*1\.", out)
    assert re.search(r"\n\s*2\.", out)


def test_write_test_result_to_file_conversational(tmp_path: Path):
    turns = [
        TurnApi(order=0, role="user", content="Hello"),
        TurnApi(
            order=1,
            role="assistant",
            content="Hi",
            toolsCalled=[ToolCall(name="x")],
        ),
    ]
    tr = EvalTestResult(
        name="demo",
        success=True,
        input=None,
        conversational=True,
        metrics_data=[],
        turns=turns,
    )

    write_test_result_to_file(tr, RunResultDisplay.ALL, str(tmp_path))

    # look only at files (skip .deepeval/ and any other dirs)
    text = None
    for f in tmp_path.iterdir():
        if not f.is_file():
            continue
        try:
            content = f.read_text()
        except UnicodeDecodeError:
            continue  # skip any non-text files, just in case
        if "For conversational test case:" in content:
            text = content
            break

    assert text, "Couldn't find conversational output file"
    assert "  Turns:" in text
    assert "  | tools: x" in text
