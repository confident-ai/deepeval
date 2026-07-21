"""Browser authentication and CLI-native onboarding for `deepeval login`.

Device-code style pairing (RFC 8628 shaped), with no localhost callback
server. The CLI never handles credentials and no secrets ever transit
localhost, so the flow works over SSH / containers / Colab and is unaffected
by Chrome's Local Network Access permission prompt.

Flow:

1. The CLI asks the backend to create a short-lived login session:

       POST {api_base}/cli/auth/sessions
       -> 200 {"success": true,
               "data": {"userCode": "ABCD-1234",
                        "deviceCode": "<opaque high-entropy token>",
                        "expiresIn": 600,
                        "interval": 3}}

   `userCode` is short and user-visible (shown in the terminal and in the
   browser so the user can confirm they're pairing the right session).
   `deviceCode` is a secret with >= 128 bits of entropy; it never leaves the
   CLI and is the only thing that can claim the API key.

2. The CLI opens the browser at the returned verification URL. The browser
   handles authentication and explicit authorization only; onboarding remains
   in the terminal.

3. The CLI polls the token endpoint with the secret device code:

       POST {api_base}/cli/auth/sessions/token
            {"deviceCode": "<token>"}
       -> pending:   {"success": true, "data": {"status": "pending"}}
       -> authenticated: {"success": true,
                      "data": {"status": "authenticated",
                               "setupToken": "<short-lived token>",
                               "email": "user@example.com"}}
       -> expired / consumed / unknown: 404, or
                     {"success": false, "error": "..."}

4. The CLI fetches onboarding context and completes setup using the short-lived
   setup token. The completion response contains a newly minted project API
   key, and consumes the login session.
"""

import sys
import time
import webbrowser
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
import typer
from pydantic import ValidationError
from rich import print

import deepeval
from deepeval.cli.auth.api import (
    CliAuthorization,
    CliOnboardingContext,
    DynamicNewUserOnboardingRequest,
    ExistingProjectKeyRequest,
    DevicePairing,
)
from deepeval.cli.utils import with_utm
from deepeval.confident.api import get_base_api_url
from deepeval.telemetry import set_logged_in_with

CREATE_PAIRING_ENDPOINT = "/cli/auth/sessions"
PAIRING_TOKEN_ENDPOINT = "/cli/auth/sessions/token"
ONBOARDING_ENDPOINT = "/cli/onboarding"
ONBOARDING_COMPLETE_ENDPOINT = "/cli/onboarding/complete"

REQUEST_TIMEOUT_SECONDS = 10
# Completion creates the organization/project in one backend transaction,
# which can take longer than a normal request. Retries are safe because the
# request carries an idempotency key.
COMPLETE_TIMEOUT_SECONDS = 30
COMPLETE_MAX_ATTEMPTS = 3


class AuthFlowError(Exception):
    """Raised when the browser login flow cannot complete."""


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


def _questionary_style():
    import questionary

    return questionary.Style(
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


def prompt_text(message: str, default: Optional[str] = None) -> str:
    """Styled text input matching `prompt_select`, with a plain fallback."""
    if _is_interactive():
        try:
            import questionary

            answer = questionary.text(
                message,
                default=default or "",
                qmark="?",
                style=_questionary_style(),
            ).ask()
            # `ask()` returns None when the user aborts (e.g. Ctrl-C).
            if answer is None:
                raise AuthFlowError("Input cancelled.")
            return answer
        except AuthFlowError:
            raise
        except Exception:
            # Any questionary/terminal issue: degrade to the plain prompt.
            pass

    return typer.prompt(message, default=default)


def prompt_select(message: str, choices: Sequence[Tuple[str, Any]]) -> Any:
    """Arrow-key selection when the terminal supports it, numbered fallback
    otherwise."""
    if _is_interactive():
        try:
            import questionary

            style = _questionary_style()
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


def prompt_checkbox(
    message: str,
    choices: Sequence[Tuple[str, Any]],
    min_selections: int = 1,
) -> List[Any]:
    """Arrow-key multi-selection with a comma-separated numbered fallback."""
    if _is_interactive():
        try:
            import questionary

            answer = questionary.checkbox(
                message,
                choices=[
                    questionary.Choice(title=label, value=value)
                    for label, value in choices
                ],
                qmark="?",
                pointer="❯",
                instruction="(space to select, enter to continue)",
                style=_questionary_style(),
                validate=lambda values: len(values) >= min_selections
                or f"Select at least {min_selections} option(s).",
            ).ask()
            if answer is None:
                raise AuthFlowError("Selection cancelled.")
            return list(answer)
        except AuthFlowError:
            raise
        except Exception:
            # Any questionary/terminal issue: degrade to the numbered prompt.
            pass

    print(message)
    for index, (label, _) in enumerate(choices, start=1):
        print(f"  {index}. {label}")
    while True:
        if min_selections == 0:
            raw = typer.prompt(
                "Enter numbers (comma-separated), or leave blank",
                default="",
                show_default=False,
            )
            if not raw.strip():
                return []
        else:
            raw = typer.prompt("Enter one or more numbers (comma-separated)")
        try:
            selected = [int(value.strip()) for value in raw.split(",")]
        except (AttributeError, TypeError, ValueError):
            print("❌ Enter valid numbers separated by commas.")
            continue
        if (
            len(selected) >= min_selections
            and len(selected) == len(set(selected))
            and all(1 <= value <= len(choices) for value in selected)
        ):
            return [choices[value - 1][1] for value in selected]
        print(
            f"❌ Select at least {min_selections} unique number(s) between "
            f"1 and {len(choices)}, separated by commas."
        )


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
    try:
        return DevicePairing.model_validate(data)
    except ValidationError:
        raise AuthFlowError(
            f"POST {url} did not return the required login session fields: "
            f"{res.text[:200]!r}"
        )


def _poll_once(
    device_code: str,
) -> Optional[CliAuthorization]:
    """One poll of the token endpoint.

    Returns setup authorization when browser auth completes, None while pending
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
    if status == "authenticated":
        try:
            return CliAuthorization.model_validate(data)
        except ValidationError:
            raise AuthFlowError(
                "Browser authentication completed but the server did not "
                "return a setup token."
            )
    raise AuthFlowError(
        f"The pairing is no longer valid (status: {status or 'unknown'}). "
        "Run `deepeval login` again."
    )


def browser_pairing_login() -> Optional[CliAuthorization]:
    """Authenticate the user in the browser and return CLI setup access.

    Returns a short-lived setup authorization, or None if the flow was aborted
    / timed out / unavailable.
    """
    try:
        pairing = create_pairing()
    except AuthFlowError:
        print(
            "\n⚠️  Unexpected error — seems like browser login isn't "
            "available right now."
        )
        return None

    login_url = with_utm(
        pairing.verification_url,
        medium="cli",
        content="login_pair_browser_open",
    )
    fallback_url = with_utm(
        pairing.verification_url,
        medium="cli",
        content="login_pair_fallback_link",
    )

    print("\n🌐 Opening your browser — confirm this pairing code there:")
    print(f"\n    [bold green]{pairing.user_code}[/bold green]\n")
    webbrowser.open(login_url)
    print(
        f"[dim]Browser didn't open? Use [link={fallback_url}]{fallback_url}[/link][/dim]"
    )
    print(
        "[dim]Waiting for the browser... "
        "press Ctrl+C to paste an API key instead.[/dim]"
    )

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
            if completed.email:
                set_logged_in_with(completed.email)
            return completed
        print("\n⌛ Timed out waiting for the browser.")
        return None
    except KeyboardInterrupt:
        print()
        return None


def get_cli_onboarding_context(setup_token: str) -> CliOnboardingContext:
    url = f"{get_base_api_url()}{ONBOARDING_ENDPOINT}"
    try:
        res = requests.get(
            url,
            headers={
                **_request_headers(),
                "Authorization": f"Bearer {setup_token}",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise AuthFlowError(f"Could not reach GET {url} ({e}).")
    if res.status_code != 200:
        raise AuthFlowError(
            f"GET {url} failed with HTTP {res.status_code}: {res.text[:200]!r}"
        )
    try:
        data = res.json()
    except ValueError:
        raise AuthFlowError(f"GET {url} did not return JSON.")
    try:
        return CliOnboardingContext.model_validate(data)
    except ValidationError:
        raise AuthFlowError(
            f"GET {url} returned an unexpected onboarding payload."
        )


def complete_cli_onboarding(
    setup_token: str,
    request: Union[
        DynamicNewUserOnboardingRequest,
        ExistingProjectKeyRequest,
    ],
    idempotency_key: str,
) -> str:
    url = f"{get_base_api_url()}{ONBOARDING_COMPLETE_ENDPOINT}"
    res = None
    last_error: Optional[requests.RequestException] = None
    for attempt in range(COMPLETE_MAX_ATTEMPTS):
        retryable = attempt < COMPLETE_MAX_ATTEMPTS - 1
        try:
            res = requests.post(
                url,
                headers={
                    **_request_headers(),
                    "Authorization": f"Bearer {setup_token}",
                    "Idempotency-Key": idempotency_key,
                },
                json=request.to_payload(),
                timeout=COMPLETE_TIMEOUT_SECONDS,
            )
        except requests.RequestException as e:
            last_error = e
            if retryable:
                print("Still working — retrying...")
                time.sleep(2)
            continue
        # A previous (timed-out) attempt may still hold the completion lock
        # server-side; give it a moment and retry with the same key.
        if res.status_code == 409 and "in progress" in res.text and retryable:
            time.sleep(3)
            continue
        break
    if res is None:
        raise AuthFlowError(f"Could not reach POST {url} ({last_error}).")
    if res.status_code != 200:
        raise AuthFlowError(
            f"POST {url} failed with HTTP {res.status_code}: "
            f"{res.text[:200]!r}"
        )
    try:
        data = res.json()
    except ValueError:
        raise AuthFlowError(f"POST {url} did not return JSON.")
    if not isinstance(data, dict) or not data.get("apiKey"):
        raise AuthFlowError(
            "CLI onboarding completed without returning an API key."
        )
    return str(data["apiKey"])
