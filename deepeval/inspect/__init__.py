"""TUI for inspecting `test_run_*.json` files. CLI entry: `deepeval inspect [PATH]`."""


def run_inspect(path: str) -> None:
    # Lazy imports keep `import deepeval.inspect` free of Textual /
    # pyperclip until the user actually invokes the TUI.
    from deepeval.inspect.app import InspectApp
    from deepeval.inspect.loader import load_test_run

    traces = load_test_run(path)
    InspectApp(traces=traces, source_path=path).run()
