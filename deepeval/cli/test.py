import pytest
import typer
import os
from typing_extensions import Annotated
from typing import Optional
from deepeval.test_run import test_run_manager, TEMP_FILE_NAME
from deepeval.utils import delete_file_if_exists

try:
    from rich import print
except Exception as e:
    pass


app = typer.Typer(name="test")


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
