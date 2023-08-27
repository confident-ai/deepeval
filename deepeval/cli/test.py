import pytest
import typer
from .cli_key_handler import set_env_vars
from ..metrics.overall_score import assert_overall_score

try:
    from rich import print
except Exception as e:
    pass

app = typer.Typer(name="test")


def sample():
    set_env_vars()
    print("Sending sample test results...")
    print("If this is your first time running these models, it may take a while.")
    try:
        query = "How does photosynthesis work?"
        output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
        expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
        context = "Biology"

        assert_overall_score(query, output, expected_output, context)

    except AssertionError as e:
        pass
    try:
        query = "What is the capital of France?"
        output = "The capital of France is Paris."
        expected_output = "The capital of France is Paris."
        context = "Geography"

        assert_overall_score(query, output, expected_output, context)

    except AssertionError as e:
        pass
    try:
        query = "What are the major components of a cell?"
        output = "Cells have many major components, including the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
        expected_output = "Cells have several major components, such as the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
        context = "Biology"
        minimum_score = 0.8  # Adjusting the minimum score threshold

        assert_overall_score(query, output, expected_output, context, minimum_score)

    except AssertionError as e:
        pass

    try:
        query = "What is the capital of Japan?"
        output = "The largest city in Japan is Tokyo."
        expected_output = "The capital of Japan is Tokyo."
        context = "Geography"

        assert_overall_score(query, output, expected_output, context)
    except AssertionError as e:
        pass

    try:
        query = "Explain the theory of relativity."
        output = "Einstein's theory of relativity is famous."
        expected_output = "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity."
        context = "Physics"

        assert_overall_score(query, output, expected_output, context)
    except AssertionError as e:
        pass


def check_if_legit_file(test_file: str):
    if test_file.endswith(".py"):
        if not test_file.startswith("test_"):
            raise ValueError("Test will not run. Please ensure the `test_` prefix.")


@app.command()
def run(test_file_or_directory: str, exit_on_first_failure: bool = False):
    """Run a test"""
    if test_file_or_directory == "sample":
        sample()
        print(
            "You can generate a sample test using [bold]deepeval test generate[/bold]."
        )
        retcode = 0
    if exit_on_first_failure:
        retcode = pytest.main(["-x", "-k", test_file_or_directory])
    else:
        retcode = pytest.main(["-k", test_file_or_directory])
    print("✅ Tests finished! View results on https://app.confident-ai.com/")
    return retcode


@app.command()
def generate(sample_file: str = "test_sample.py"):
    with open(sample_file, "w") as f:
        f.write(
            """from deepeval.metrics.overall_score import assert_overall_score

def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    assert_overall_score(query, output, expected_output, context)


def test_1():
    query = "What is the capital of France?"
    output = "The capital of France is Paris."
    expected_output = "The capital of France is Paris."
    context = "Geography"

    assert_overall_score(query, output, expected_output, context)

def test_2():
    query = "What are the major components of a cell?"
    output = "Cells have many major components, including the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
    expected_output = "Cells have several major components, such as the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
    context = "Biology"
    minimum_score = 0.8  # Adjusting the minimum score threshold

    assert_overall_score(query, output, expected_output, context, minimum_score)


def test_3():
    query = "What is the capital of Japan?"
    output = "The largest city in Japan is Tokyo."
    expected_output = "The capital of Japan is Tokyo."
    context = "Geography"

    assert_overall_score(query, output, expected_output, context)

def test_4():
    query = "Explain the theory of relativity."
    output = "Einstein's theory of relativity is famous."
    expected_output = "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity."
    context = "Physics"

    assert_overall_score(query, output, expected_output, context)
"""
        )

    print(f"✨ Done! Now run: [bold]deepeval test run {sample_file}[/bold]")
    print(
        "You can generate more tests in the future in our documentation at https://docs.confident-ai.com/docs"
    )
