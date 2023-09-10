import pytest
import typer
from .cli_key_handler import set_env_vars
from ..metrics.overall_score import assert_overall_score

try:
    from rich import print
    from rich.progress import Progress, SpinnerColumn, TextColumn
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
def run(
    test_file_or_directory: str,
    exit_on_first_failure: bool = False,
    verbose: bool = False,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
):
    """Run a test"""
    pytest_args = ["-k", test_file_or_directory]
    if exit_on_first_failure:
        pytest_args.insert(0, "-x")
    pytest_args.extend(
        [
            "--verbose" if verbose else "--quiet",
            f"--color={color}",
            f"--durations={durations}",
            # f"--cov={cov}",
            # f"--cov-report={cov_report}",
            "--pdb" if pdb else "",
        ]
    )
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
    print("✅ Tests finished! View results on https://app.confident-ai.com/")
    return retcode


@app.command()
def generate(output_file: str = "test_sample.py"):
    with open(output_file, "w") as f:
        f.write(
            """from deepeval.metrics.overall_score import assert_overall_score
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric
from deepeval.metrics.factual_consistency import FactualConsistencyMetric


def test_0():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    assert_overall_score(query, output, expected_output, context)


def test_1():
    # Check to make sure it is relevant
    query = "What is the capital of France?"
    output = "The capital of France is Paris."
    metric = AnswerRelevancyMetric(minimum_score=0.5)
    assert metric.is_successful()


def test_2():
    # Check to make sure it is factually consistent
    output = "Cells have many major components, including the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
    context = "Biology"
    minimum_score = 0.8  # Adjusting the minimum score threshold
    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    metric.measure(output, context)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def test_3():
    query = "What is the capital of Japan?"
    output = "The capital of Japan is Tokyo."
    expected_output = "The capital of Japan is Tokyo."
    context = "Geography"

    assert_overall_score(query, output, expected_output, context)


def test_4():
    query = "Explain the theory of relativity."
    output = "Einstein's theory of relativity is famous."
    expected_output = "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity."
    context = "Physics"

    assert_overall_score(query, output, expected_output, context)


def test_5():
    output = "China has the largest population in the world."
    context = "Geography"
    minimum_score = 0.9  # Adjusting the minimum score threshold
    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    metric.measure(output, context)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def test_6():
    query = "How does photosynthesis work?"
    output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigment."
    expected_output = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food with the help of chlorophyll pigment."
    context = "Biology"

    assert_overall_score(query, output, expected_output, context)


def test_7():
    # Check to make sure it is relevant
    query = "What is the capital of France?"
    output = "The capital of France is Paris."
    metric = AnswerRelevancyMetric(minimum_score=0.5)
    metric.measure(query, output)
    assert metric.is_successful()


def test_8():
    # Check to make sure it is factually consistent
    output = "Cells have many major components, including the cell membrane, nucleus, mitochondria, and endoplasmic reticulum."
    context = "Biology"
    minimum_score = 0.8  # Adjusting the minimum score threshold
    metric = FactualConsistencyMetric(minimum_score=minimum_score)
    score = metric.measure(output, context)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def test_9():
    output = "The largest city in Japan is Tokyo."
    expected_output = "The capital of Japan is Tokyo."

    metric = ConceptualSimilarityMetric(minimum_score=0.5)
    metric.measure(output, expected_output)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def test_10():
    output = "Einstein's theory of relativity is famous."
    expected_output = "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity."

    metric = ConceptualSimilarityMetric(minimum_score=0.5)
    metric.measure(output, expected_output)
    assert metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."


def test_11():
    output = "China has the largest population in the world."
    expected_output = "Einstein's theory of relativity revolutionized our understanding of space, time, and gravity."
    metric = ConceptualSimilarityMetric(minimum_score=0.5)
    metric.measure(output, expected_output)
    assert not metric.is_successful(), metric.__class__.__name__ + " was unsuccessful."
"""
        )

    print(f"✨ Done! Now run: [bold]deepeval test run {output_file}[/bold]")
    print(
        "You can generate more tests in the future in our documentation at https://docs.confident-ai.com/docs"
    )
