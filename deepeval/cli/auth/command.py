"""`deepeval login` / `deepeval logout` Typer commands."""

from typing import Optional

import typer
from rich import print

from deepeval.cli.auth.flow import browser_pairing_login, prompt_select
from deepeval.cli.diagnose import resolve_setting_source
from deepeval.cli.dotenv_handler import DotenvHandler
from deepeval.cli.utils import (
    PROD,
    WWW,
    coerce_blank_to_none,
    handle_save_result,
    render_login_message,
    with_utm,
)
from deepeval.config.settings import dotenv_search_paths, get_settings
from deepeval.config.utils import read_dotenv_file
from deepeval.telemetry import capture_login_event
from deepeval.test_run.test_run import LATEST_TEST_RUN_FILE_PATH
from deepeval.utils import delete_file_if_exists

LOGIN_HELP = (
    "Log in to Confident AI. Opens the platform in your browser to sign in "
    "(or create an account) and pick a project; the project API key is "
    "paired back automatically. "
    f"Get a project API key from {with_utm(PROD, medium='cli', content='login_help_text')}. "
    "The key is saved to your environment variables, typically .env.local, "
    "unless a different path is provided with --save."
)


def _prompt_paste_api_key() -> str:
    while True:
        api_key = coerce_blank_to_none(
            typer.prompt("🔐 Enter your project API key", hide_input=True)
        )
        if api_key:
            return api_key
        print("❌ Project API key cannot be empty. Please try again.\n")


def _resolve_login_key() -> str:
    render_login_message()

    method = prompt_select(
        "How would you like to log in?",
        [
            ("Log in via your browser", "browser"),
            ("Paste a project API key", "paste"),
        ],
    )

    if method == "paste":
        return _prompt_paste_api_key()

    key = browser_pairing_login()
    if key:
        return key
    print("\nNo problem — paste a project API key from the platform instead.")
    return _prompt_paste_api_key()


def login_command(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to persist settings. Format: dotenv[:path]. Defaults to .env.local. If omitted, login still writes to .env.local.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Log in non-interactively with a project API key instead of the prompts.",
    ),
):
    api_key = coerce_blank_to_none(api_key)

    with capture_login_event() as span:
        completed = False
        try:
            # Resolve the key from the CLI flag or the browser pairing flow.
            if api_key is not None:
                key = api_key
            else:
                key = _resolve_login_key()

            settings = get_settings()
            save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"
            with settings.edit(save=save) as edit_ctx:
                settings.CONFIDENT_API_KEY = key

            handled, path, updated = edit_ctx.result

            if updated:
                if not handled and save is not None:
                    # invalid --save format (unsupported)
                    print(
                        "Unsupported --save option. Use --save=dotenv[:path]."
                    )
                elif path:
                    # persisted to a file
                    print(
                        f"Saved environment variables to {path} (ensure it's git-ignored)."
                    )

            completed = True
            print(
                "\n🎉🥳 Congratulations! You've successfully logged in! :raising_hands:"
            )
            quickstart_url = with_utm(
                f"{WWW}/docs/llm-evaluation/quickstart",
                medium="cli",
                content="login_success_quickstart",
            )
            print(
                "You're now using DeepEval with [rgb(106,0,255)]Confident AI[/rgb(106,0,255)]. "
                "Follow our quickstart tutorial here: "
                f"[bold][link={quickstart_url}]{quickstart_url}[/link][/bold]"
            )
        except Exception as e:
            completed = False
            print(f"Login failed: {e}")
        finally:
            if getattr(span, "set_attribute", None):
                span.set_attribute("completed", completed)


def logout_command(
    save: Optional[str] = typer.Option(
        None,
        "-s",
        "--save",
        help="Where to remove the saved key from. Use format dotenv[:path]. If omitted, uses DEEPEVAL_DEFAULT_SAVE or .env.local. The JSON keystore is always cleared.",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress printing to the terminal (useful for CI).",
    ),
):
    """
    Log out of Confident AI.

    Behavior:
    - Always clears the Confident API key from the JSON keystore and process env.
    - Removes the key from every dotenv file deepeval auto-loads (.env,
      .env.{APP_ENV}, .env.local), plus the --save target if one is given.
    - If the key is exported by the shell itself, deepeval cannot unset it;
      a warning with the fix is printed instead of a success message.
    """
    settings = get_settings()

    # Capture provenance before clearing anything: once files are wiped we
    # can no longer tell a shell export apart from a file-loaded value.
    key_source = resolve_setting_source("CONFIDENT_API_KEY")

    save = save or settings.DEEPEVAL_DEFAULT_SAVE or "dotenv:.env.local"
    with settings.edit(save=save) as edit_ctx:
        settings.CONFIDENT_API_KEY = None

    handled, path, updated = edit_ctx.result

    # The --save target is a single file; also sweep the rest of the dotenv
    # search path so a lower-precedence file (e.g. .env) can't silently log
    # the user back in on the next run.
    for dotenv_path in dotenv_search_paths():
        if dotenv_path.is_file() and "CONFIDENT_API_KEY" in read_dotenv_file(
            dotenv_path
        ):
            DotenvHandler(dotenv_path).unset(["CONFIDENT_API_KEY"])
            if not quiet:
                print(f"Removed Confident AI key(s) from {dotenv_path}.")

    shell_export = key_source == "process environment"

    if (
        handle_save_result(
            handled=handled,
            path=path,
            updates=updated,
            save=save,
            quiet=quiet,
            updated_msg="Removed Confident AI key(s) from {path}.",
            tip_msg=None,
        )
        and not shell_export
    ):
        print("\n🎉🥳 You've successfully logged out! :raising_hands: ")

    if shell_export and not quiet:
        print(
            "\n[yellow]⚠  CONFIDENT_API_KEY is exported by your shell, which "
            "deepeval cannot unset — this terminal will still be logged "
            "in.[/yellow]"
        )
        print(
            "   Finish logging out with: "
            "[bold cyan]unset CONFIDENT_API_KEY[/bold cyan] "
            "[dim](and remove it from your shell profile if it's set "
            "there)[/dim]"
        )

    delete_file_if_exists(LATEST_TEST_RUN_FILE_PATH)
