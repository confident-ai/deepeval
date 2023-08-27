import typer
from .cli_key_handler import set_env_vars
from ..metrics.factual_consistency import assert_factual_consistency

try:
    from rich import print
except Exception as e:
    pass

app = typer.Typer(name="test")


@app.command()
def sample():
    set_env_vars()
    print("Sending sample test results...")
    print("If this is your first time running these models, it may take a while.")
    try:
        assert_factual_consistency(
            output="Python is an interpreted high-level programming language.",
            context="Python is a programming language.",
        )
    except AssertionError as e:
        pass
    try:
        assert_factual_consistency(
            output="Python is an interpreted high-level programming language.",
            context="Python is a programming language.",
        )
    except AssertionError as e:
        pass
    try:
        assert_factual_consistency(
            output="Water boils at 100 degrees Fahrenheit.",
            context="Water boils at 100 degrees Celsius.",
        )
    except AssertionError as e:
        pass
    print("✅ Tests finished! View results on https://app.confident-ai.com/")
