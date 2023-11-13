import pytest
import typer
import os
from typing_extensions import Annotated
from deepeval.metrics.overall_score import assert_overall_score
from .cli_key_handler import set_env_vars
from typing import Optional
from deepeval.test_run import test_run_manager, TEMP_FILE_NAME
from deepeval.utils import delete_file_if_exists

try:
    from rich import print
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


def check_if_valid_file(test_file_or_directory: str):
    if "::" in test_file_or_directory:
        test_file_or_directory, test_case = test_file_or_directory.split("::")
    if os.path.isfile(test_file_or_directory):
        if test_file_or_directory.endswith(".py"):
            if not os.path.basename(test_file_or_directory).startswith("test_"):
                raise ValueError(
                    "Test will not run. Please ensure the file starts with `test_` prefix."
                )
    elif os.path.isdir(test_file_or_directory):
        return
    else:
        raise ValueError(
            "Provided path is neither a valid file nor a directory."
        )


@app.command()
def run(
    test_file_or_directory: str,
    verbose: bool = True,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
    exit_on_first_failure: Annotated[
        bool, typer.Option("--exit-on-first-failure", "-x/-X")
    ] = False,
    show_warnings: Annotated[
        bool, typer.Option("--show-warnings", "-w/-W")
    ] = False,
    num_processes: Optional[int] = typer.Option(
        None,
        "--num-processes",
        "-n",
        help="Number of processes to use with pytest",
    ),
):
    """Run a test"""
    delete_file_if_exists(TEMP_FILE_NAME)
    check_if_valid_file(test_file_or_directory)
    pytest_args = [test_file_or_directory]
    if exit_on_first_failure:
        pytest_args.insert(0, "-x")

    pytest_args.extend(
        [
            "--verbose" if verbose else "--quiet",
            f"--color={color}",
            f"--durations={durations}",
            "-s",
        ]
    )
    if pdb:
        pytest_args.append("--pdb")
    if not show_warnings:
        pytest_args.append("--disable-warnings")
    if num_processes is not None:
        pytest_args.extend(["-n", str(num_processes)])

    # Add the deepeval plugin file to pytest arguments
    pytest_args.extend(["-p", "plugins"])

    retcode = pytest.main(pytest_args)

    test_run_manager.wrap_up_test_run()
    return retcode


@app.command()
def generate(output_file: str = "test_sample.py"):
    with open(
        os.path.join(os.path.dirname(__file__), "../test_quickstart.py"),
        "r",
    ) as f_in:
        with open(output_file, "w") as f_out:
            f_out.write(f_in.read())
    print(f"✨ Done! Now run: [bold]deepeval test run {output_file}[/bold]")
    print(
        "You can generate more tests in the future in our documentation at https://docs.confident-ai.com/docs"
    )
