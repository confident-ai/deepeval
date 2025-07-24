import os
import sys
import subprocess
import tempfile
import unittest
import shutil
import pytest


def test_import_deepeval_with_only_required_dependencies():
    """
    Test that deepeval can be imported without errors using only the
    dependencies specified in pyproject.toml.
    
    This test creates a temporary virtual environment, installs deepeval with
    its dependencies from pyproject.toml, and then runs a simple script that
    imports deepeval and its core submodules. If any ImportError occurs during
    this process, the test fails.
    
    This ensures that deepeval properly handles optional dependencies by wrapping
    them in try-except blocks, allowing the package to be used even when only the
    core dependencies are installed.
    """
    # Skip this test during CI runs, as it's meant for local development verification
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping virtual environment test in CI")

    # Create a temporary directory to hold our virtual environment
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_dir = os.path.join(tmpdir, "venv")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # Create the virtual environment
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", venv_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Determine the path to the Python executable in the virtual environment
            if os.name == "nt":  # Windows
                venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
            else:  # Unix-like systems (Linux, macOS)
                venv_python = os.path.join(venv_dir, "bin", "python")
            
            # Install deepeval using pip in the virtual environment
            subprocess.run(
                [venv_python, "-m", "pip", "install", "-e", project_root],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Create a temporary script to test imports
            test_script = os.path.join(tmpdir, "test_imports.py")
            with open(test_script, "w") as f:
                f.write("""\
import sys

try:
    import deepeval
    print("Successfully imported deepeval")
    
    # Try importing core submodules
    core_submodules = [
        "deepeval.metrics",
        "deepeval.models",
        "deepeval.dataset",
        "deepeval.evaluate",
        "deepeval.test_case",
    ]
    
    for submodule in core_submodules:
        __import__(submodule)
        print(f"Successfully imported {submodule}")
    
    sys.exit(0)
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
""")
            
            # Run the test script in the virtual environment
            result = subprocess.run(
                [venv_python, test_script],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Check if the script ran successfully
            assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"
            
        except subprocess.CalledProcessError as e:
            assert False, f"Failed to set up or run test in virtual environment: {e}\nstdout: {e.stdout.decode() if e.stdout else ''}\nstderr: {e.stderr.decode() if e.stderr else ''}"


# These tests run in the current environment and are useful for quick checks
# but don't guarantee that only pyproject.toml dependencies are being used

def test_basic_import_deepeval_succeeds():
    """Basic test that deepeval can be imported without errors in the current environment."""
    try:
        import deepeval
        assert True
    except ImportError as e:
        assert False, f"ImportError when importing deepeval: {e}"


def test_basic_import_submodules_succeeds():
    """Basic test that deepeval submodules can be imported in the current environment."""
    core_submodules = [
        "deepeval.metrics",
        "deepeval.models",
        "deepeval.dataset",
        "deepeval.evaluate",
        "deepeval.test_case",
    ]
    
    for submodule in core_submodules:
        try:
            __import__(submodule)
            assert True
        except ImportError as e:
            assert False, f"ImportError when importing {submodule}: {e}"