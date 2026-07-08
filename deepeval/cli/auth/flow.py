"""Browser-based login flow for `deepeval login`.

Device-code style pairing (RFC 8628 shaped), with no localhost callback
server. The CLI never handles credentials and no secrets ever transit
localhost, so the flow works over SSH / containers / Colab and is unaffected
by Chrome's Local Network Access permission prompt.

Flow:

1. The CLI asks the backend to create a pairing:

       POST {api_base}/v1/cli/login/pairings
       -> 200 {"success": true,
               "data": {"userCode": "ABCD-1234",
                        "deviceCode": "<opaque high-entropy token>",
                        "expiresIn": 600,
                        "interval": 3}}

   `userCode` is short and user-visible (shown in the terminal and in the
   browser so the user can confirm they're pairing the right session).
   `deviceCode` is a secret with >= 128 bits of entropy; it never leaves the
   CLI and is the only thing that can claim the API key.

2. The CLI opens the browser at `{PROD}/auth/login?code=<userCode>`. The web
   app carries the code through login / signup / onboarding, lets the user
   pick a project, then attaches a project API key to the pairing
   server-side (e.g. an authenticated
   `POST /cli/login/pairings/:userCode/attach {projectId}` from the web app).

3. The CLI polls the token endpoint with the secret device code:

       POST {api_base}/v1/cli/login/pairings/token
            {"deviceCode": "<token>"}
       -> pending:   {"success": true, "data": {"status": "pending"}}
       -> completed: {"success": true,
                      "data": {"status": "completed",
                               "apiKey": "confident_us_...",
                               "email": "user@example.com"}}
       -> expired / consumed / unknown: 404, or
                     {"success": false, "error": "..."}

   The completed response must be one-time: the first successful claim
   consumes the pairing.
"""

import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import requests
import typer
from rich import print

import deepeval
from deepeval.cli.utils import PROD, with_utm
from deepeval.confident.api import get_base_api_url
from deepeval.telemetry import set_logged_in_with

CREATE_PAIRING_ENDPOINT = "/v1/cli/login/pairings"
PAIRING_TOKEN_ENDPOINT = "/v1/cli/login/pairings/token"

DEFAULT_EXPIRES_IN_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 3
REQUEST_TIMEOUT_SECONDS = 10


class AuthFlowError(Exception):
    """Raised when the browser login flow cannot complete."""


@dataclass
class DevicePairing:
    user_code: str
    device_code: str
    expires_in: int
    interval: int


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


CONFIDENT_PURPLE = "#6a00ff"


def _fill_radio_indicator(question: Any) -> None:
    """Make the select behave like a radio group: ● on the pointed row, ○ on
    the rest.

    questionary's `use_indicator` renders a hollow circle on every row in
    select mode (the filled one is reserved for checkbox mode, driven by
    `selected_options`, which select never populates). Sync `selected_options`
    with the pointed row at render time so the active option gets the filled
    dot.
    """
    from questionary.prompts.common import InquirerControl

    for control in question.application.layout.find_all_controls():
        if isinstance(control, InquirerControl):
            original = control.text

            def patched(control=control, original=original):
                control.selected_options = [control.get_pointed_at().value]
                return original()

            control.text = patched
            return


def prompt_select(message: str, choices: Sequence[Tuple[str, Any]]) -> Any:
    """Arrow-key selection when the terminal supports it, numbered fallback
    otherwise."""
    if _is_interactive():
        try:
            import questionary

            style = questionary.Style(
                [
                    ("qmark", f"fg:{CONFIDENT_PURPLE} bold"),
                    ("question", "bold"),
                    ("instruction", "fg:#767676"),
                    ("pointer", f"fg:{CONFIDENT_PURPLE} bold"),
                    ("highlighted", f"fg:{CONFIDENT_PURPLE} bold"),
                    # prompt_toolkit's built-in style renders "selected" in
                    # reverse video (fg/bg swapped); noreverse disables that.
                    ("selected", f"fg:{CONFIDENT_PURPLE} bold noreverse"),
                    ("answer", f"fg:{CONFIDENT_PURPLE} bold"),
                ]
            )
            question = questionary.select(
                message,
                choices=[
                    questionary.Choice(title=label, value=value)
                    for label, value in choices
                ],
                qmark="?",
                pointer="❯",
                use_indicator=True,
                instruction="(arrow keys)",
                style=style,
            )
            _fill_radio_indicator(question)
            answer = question.ask()
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


def _request_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-DeepEval-Version": deepeval.__version__,
    }


def _unwrap(res: requests.Response) -> Dict[str, Any]:
    """Unwrap the standard `{success, data, error}` API envelope. Tolerates a
    bare payload so the backend can be iterated on without breaking the CLI."""
    try:
        payload = res.json()
    except ValueError:
        raise AuthFlowError(
            f"{res.request.method} {res.url} did not return JSON: "
            f"{res.text[:200]!r}"
        )
    if not isinstance(payload, dict):
        raise AuthFlowError(
            f"{res.request.method} {res.url} returned an unexpected payload: "
            f"{res.text[:200]!r}"
        )
    if "success" in payload:
        if not payload.get("success"):
            raise AuthFlowError(
                payload.get("error") or "The pairing request failed."
            )
        data = payload.get("data")
        return data if isinstance(data, dict) else {}
    return payload


def create_pairing() -> DevicePairing:
    url = f"{get_base_api_url()}{CREATE_PAIRING_ENDPOINT}"
    try:
        res = requests.post(
            url,
            headers=_request_headers(),
            json={},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise AuthFlowError(f"Could not reach POST {url} ({e}).")

    if res.status_code == 404:
        raise AuthFlowError(
            f"The backend does not support browser login yet "
            f"(404 from POST {url})."
        )
    if res.status_code != 200:
        raise AuthFlowError(
            f"POST {url} failed with HTTP {res.status_code}: {res.text[:200]!r}"
        )

    data = _unwrap(res)
    user_code = data.get("userCode")
    device_code = data.get("deviceCode")
    if not user_code or not device_code:
        raise AuthFlowError(
            f"POST {url} did not return userCode/deviceCode: {res.text[:200]!r}"
        )
    return DevicePairing(
        user_code=str(user_code),
        device_code=str(device_code),
        expires_in=int(data.get("expiresIn") or DEFAULT_EXPIRES_IN_SECONDS),
        interval=int(data.get("interval") or DEFAULT_POLL_INTERVAL_SECONDS),
    )


def _poll_once(
    device_code: str,
) -> Optional[Tuple[str, Optional[str]]]:
    """One poll of the token endpoint.

    Returns (api_key, email) when the pairing completes, None while pending
    (or on a transient network error), and raises AuthFlowError on terminal
    failures (expired, consumed, backend missing).
    """
    url = f"{get_base_api_url()}{PAIRING_TOKEN_ENDPOINT}"
    try:
        res = requests.post(
            url,
            headers=_request_headers(),
            json={"deviceCode": device_code},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None

    if res.status_code == 404:
        raise AuthFlowError(
            f"This pairing is no longer valid, or the backend does not "
            f"support browser login yet (404 from POST {url})."
        )
    if res.status_code != 200:
        raise AuthFlowError(
            f"POST {url} failed with HTTP {res.status_code}: {res.text[:200]!r}"
        )

    data = _unwrap(res)
    status = data.get("status")
    if status == "pending":
        return None
    if status == "completed":
        api_key = data.get("apiKey")
        if not api_key:
            raise AuthFlowError(
                "The pairing completed but the server did not return an "
                "API key."
            )
        return str(api_key), data.get("email")
    raise AuthFlowError(
        f"The pairing is no longer valid (status: {status or 'unknown'}). "
        "Run `deepeval login` again."
    )


def browser_pairing_login() -> Optional[str]:
    """Run the full browser login flow.

    Returns the project API key, or None if the flow was aborted / timed out /
    unavailable (callers should fall back to manual key entry).
    """
    try:
        pairing = create_pairing()
    except AuthFlowError as e:
        print(f"\n⚠️  Browser login isn't available right now: {e}")
        return None

    pair_path = f"{PROD}/auth/login?code={pairing.user_code}"
    login_url = with_utm(
        pair_path, medium="cli", content="login_pair_browser_open"
    )
    fallback_url = with_utm(
        pair_path, medium="cli", content="login_pair_fallback_link"
    )

    print("\n🌐 Opening your browser to finish signing in...")
    print(
        f"Your pairing code is [bold]{pairing.user_code}[/bold] — "
        "confirm it matches what the browser shows."
    )
    webbrowser.open(login_url)
    print(
        "(open this link if your browser did not open: "
        f"[link={fallback_url}]{fallback_url}[/link])"
    )
    print("Waiting for you to finish signing in in the browser...")
    print("(press Ctrl+C to paste a project API key manually instead)")

    deadline = time.monotonic() + pairing.expires_in
    try:
        while time.monotonic() < deadline:
            time.sleep(pairing.interval)
            try:
                completed = _poll_once(pairing.device_code)
            except AuthFlowError as e:
                print(f"\n⚠️  {e}")
                return None
            if completed is None:
                continue
            api_key, email = completed
            if email:
                set_logged_in_with(email)
            return api_key
        print("\n⌛ Timed out waiting for the browser.")
        return None
    except KeyboardInterrupt:
        print()
        return None
