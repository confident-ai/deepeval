"""Browser-pairing flow for `deepeval login`.

The CLI never handles credentials. It opens the platform login page in the
browser (tagged with a one-time pairing code + a local callback port), then
waits for the platform to POST a project API key back to a short-lived local
server once the user has authenticated (or signed up + onboarded) and picked a
project. Login, signup, Google, SSO, 2FA and onboarding are all handled by the
platform UI.
"""

import random
import socket
import string
import sys
import threading
import webbrowser
from typing import Any, Callable, Optional, Sequence, Tuple

import typer
from rich import print

from deepeval.cli.server import PairingResult, start_server

PAIRING_TIMEOUT_SECONDS = 600


class AuthFlowError(Exception):
    """Raised when the auth flow cannot complete."""


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def prompt_select(message: str, choices: Sequence[Tuple[str, Any]]) -> Any:
    if _is_interactive():
        try:
            import questionary

            answer = questionary.select(
                message,
                choices=[
                    questionary.Choice(title=label, value=value)
                    for label, value in choices
                ],
            ).ask()
            # `ask()` returns None when the user aborts (e.g. Ctrl-C).
            if answer is None:
                raise AuthFlowError("Selection cancelled.")
            return answer
        except AuthFlowError:
            raise
        except Exception:
            # Any questionary/terminal issue: degrade to the numbered prompt.
            pass

    print(message)
    for index, (label, _) in enumerate(choices, start=1):
        print(f"  {index}. {label}")
    while True:
        raw = typer.prompt("Enter a number")
        try:
            selected = int(raw)
        except (TypeError, ValueError):
            print("❌ Please enter a valid number.")
            continue
        if 1 <= selected <= len(choices):
            return choices[selected - 1][1]
        print(f"❌ Please enter a number between 1 and {len(choices)}.")


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _generate_pairing_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def browser_pairing_login(
    prod_url: str,
    with_utm: Callable[..., str],
) -> Optional[str]:
    result = PairingResult()
    port = _find_available_port()
    pairing_code = _generate_pairing_code()

    pairing_thread = threading.Thread(
        target=start_server,
        args=(pairing_code, port, prod_url, result),
        daemon=True,
    )
    pairing_thread.start()

    pair_path = f"{prod_url}/auth/login?code={pairing_code}&port={port}"
    login_url = with_utm(
        pair_path, medium="cli", content="login_pair_browser_open"
    )

    print("\n🌐 Opening your browser to finish signing in...")
    webbrowser.open(login_url)

    fallback_url = with_utm(
        pair_path, medium="cli", content="login_pair_fallback_link"
    )
    print(
        "(open this link if your browser did not open: "
        f"[link={fallback_url}]{fallback_url}[/link])"
    )
    print("Waiting for you to finish signing in in the browser...")
    print("(press Ctrl+C to paste an API key manually instead)")

    try:
        if result.event.wait(timeout=PAIRING_TIMEOUT_SECONDS):
            return result.api_key
        print("\n⌛ Timed out waiting for the browser.")
        return None
    except KeyboardInterrupt:
        print()
        return None
