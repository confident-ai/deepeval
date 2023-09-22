import pytest
import typer
import os
import datetime
from typing_extensions import Annotated
from ..metrics.overall_score import assert_overall_score
from .cli_key_handler import set_env_vars
from ..constants import PYTEST_RUN_ENV_VAR

try:
    from rich import print
    from rich.progress import Progress, SpinnerColumn, TextColumn
except Exception as e:
    pass


app = typer.Typer(name="test")


def sample():
    set_env_vars()
    print("Sending sample test results...")
    print(
        "If this is your first time running these models, it may take a while."
    )
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

        assert_overall_score(
            query, output, expected_output, context, minimum_score
        )

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
            raise ValueError(
                "Test will not run. Please ensure the `test_` prefix."
            )


@app.command()
def run(
    test_file_or_directory: str,
    verbose: bool = False,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
    exit_on_first_failure: Annotated[
        bool, typer.Option("--exit-on-first-failure", "-x/-X")
    ] = False,
):
    """Run a test"""
    pytest_args = ["-k", test_file_or_directory]
    if exit_on_first_failure:
        pytest_args.insert(0, "-x")

    # Generate environment variable based on current date and time
    env_var = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ[PYTEST_RUN_ENV_VAR] = env_var

    pytest_args.extend(
        [
            "--verbose" if verbose else "--quiet",
            f"--color={color}",
            f"--durations={durations}",
            "--pdb" if pdb else "",
        ]
    )
    # Add the deepeval plugin file to pytest arguments
    pytest_args.extend(["-p", "plugins"])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # progress.add_task(description="Preparing tests...", total=None)
        progress.add_task(
            description="Downloading models (may take up to 2 minutes if running for the first time)...",
            total=None,
        )
        retcode = pytest.main(pytest_args)

    # Print this if the run env var is not set
    if not os.getenv(PYTEST_RUN_ENV_VAR):
        print(
            "✅ Tests finished! If logged in, view results on https://app.confident-ai.com/"
        )
    return retcode


@app.command()
def generate(output_file: str = "test_sample.py"):
    with open(output_file, "w") as f:
        f.write(
            """import pytest
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.overall_score import assert_overall_score
from deepeval.metrics.randomscore import RandomMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test


def test_1():
    # Check to make sure it is relevant
    query = "What is the capital of France?"
    output = "The capital of France is Paris."
    metric = RandomMetric()
    # Comment this out for differne metrics/models
    # metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=output)
    assert_test(test_case, [metric])


def test_2():
    # Check to make sure it is factually consistent
    output = "Cells have many major components, including the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
    context = "Biology"
    metric = RandomMetric()
    # Comment this out for factual consistency tests
    # metric = FactualConsistencyMetric(minimum_score=0.8)
    test_case = LLMTestCase(output=output, context=context)
    assert_test(test_case, [metric])


def test_3():
    # Add a test that fails
    query = "What is the capital of Germany?"
    output = "The capital of Germany is Berlin."
    metric = RandomMetric()
    # metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=query, output=output)
    assert_test(test_case, [metric])


# This is how you can skip a test
@pytest.mark.skip(reason="Can take a while")
def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    assert_overall_score(query, output, expected_output, context)
"""
        )

    print(f"✨ Done! Now run: [bold]deepeval test run {output_file}[/bold]")
    print(
        "You can generate more tests in the future in our documentation at https://docs.confident-ai.com/docs"
    )
