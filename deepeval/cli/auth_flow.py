import random
import socket
import string
import sys
import threading
import webbrowser
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
import typer
from rich import print

from deepeval.cli.server import PairingResult, start_server

REQUEST_TIMEOUT = 30
PAIRING_TIMEOUT_SECONDS = 300


class AuthFlowError(Exception):
    """Raised when an interactive auth flow cannot complete."""


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


# --------------------------------------------------------------------------- #
# Email & password (in-terminal)
# --------------------------------------------------------------------------- #


def _auth_headers(origin: str, token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Origin": origin,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _sign_in(
    session: requests.Session,
    base_url: str,
    origin: str,
    email: str,
    password: str,
) -> Tuple[Optional[str], bool]:
    resp = session.post(
        f"{base_url}/api/auth/sign-in/email",
        json={"email": email, "password": password},
        headers=_auth_headers(origin),
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code == 401:
        raise AuthFlowError("Invalid email or password.")
    if not resp.ok:
        raise AuthFlowError(_describe_error(resp, "Sign in failed"))

    data = resp.json() if resp.content else {}
    token = resp.headers.get("set-auth-token")
    two_factor_required = bool(data.get("twoFactorRedirect"))
    return token, two_factor_required


def _verify_totp(
    session: requests.Session, base_url: str, origin: str, code: str
) -> Optional[str]:
    resp = session.post(
        f"{base_url}/api/auth/two-factor/verify-totp",
        json={"code": code},
        headers=_auth_headers(origin),
        timeout=REQUEST_TIMEOUT,
    )
    if not resp.ok:
        raise AuthFlowError(_describe_error(resp, "Invalid authentication code"))
    return resp.headers.get("set-auth-token")


def _list_projects(
    session: requests.Session, base_url: str, origin: str, token: Optional[str]
) -> List[dict]:
    resp = session.get(
        f"{base_url}/cli/projects",
        headers=_auth_headers(origin, token),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code in (401, 403):
        raise AuthFlowError("Session was not accepted while listing projects.")
    if not resp.ok:
        raise AuthFlowError(_describe_error(resp, "Could not list projects"))
    return resp.json().get("projects", [])


def _create_api_key(
    session: requests.Session,
    base_url: str,
    origin: str,
    token: Optional[str],
    project_id: str,
) -> str:
    resp = session.post(
        f"{base_url}/cli/projects/{project_id}/api-key",
        json={},
        headers=_auth_headers(origin, token),
        timeout=REQUEST_TIMEOUT,
    )
    if not resp.ok:
        raise AuthFlowError(_describe_error(resp, "Could not create API key"))
    api_key = resp.json().get("apiKey", {}).get("value")
    if not api_key:
        raise AuthFlowError("Server did not return an API key.")
    return api_key


def _describe_error(resp: requests.Response, fallback: str) -> str:
    try:
        body = resp.json()
        message = body.get("message") or body.get("error")
        if isinstance(message, dict):
            message = message.get("message")
        if message:
            return f"{fallback}: {message}"
    except ValueError:
        pass
    return f"{fallback} (HTTP {resp.status_code})."


def _select_project(projects: List[dict]) -> dict:
    if not projects:
        raise AuthFlowError(
            "You don't have access to any projects yet. Create one in the "
            "browser first, then run `deepeval login` again."
        )
    if len(projects) == 1:
        only = projects[0]
        print(f"Using your only project: [bold]{only.get('name')}[/bold]")
        return only

    return prompt_select(
        "Select a project:",
        [(project.get("name") or project["id"], project) for project in projects],
    )


def email_password_login(base_url: str, origin: str) -> str:
    session = requests.Session()

    email = typer.prompt("📧 Email")
    password = typer.prompt("🔑 Password", hide_input=True)

    token, two_factor_required = _sign_in(
        session, base_url, origin, email, password
    )

    if two_factor_required:
        print("\n🔒 Two-factor authentication is enabled on your account.")
        for _ in range(3):
            code = typer.prompt("Enter the 6-digit code from your authenticator")
            try:
                token = (
                    _verify_totp(session, base_url, origin, code.strip())
                    or token
                )
                break
            except AuthFlowError as error:
                print(f"❌ {error}")
        else:
            raise AuthFlowError("Too many invalid authentication codes.")

    projects = _list_projects(session, base_url, origin, token)
    project = _select_project(projects)
    return _create_api_key(session, base_url, origin, token, project["id"])


def email_password_signup(base_url: str, origin: str, data_center: str) -> str:
    session = requests.Session()

    name = typer.prompt("👤 Your name")
    organization_name = typer.prompt("🏢 Organization name")

    token: Optional[str] = None
    while True:
        email = typer.prompt("📧 Work email")
        password = typer.prompt(
            "🔑 Password (min 8 characters)", hide_input=True
        )

        resp = session.post(
            f"{base_url}/cli/signup",
            json={
                "email": email,
                "password": password,
                "name": name,
                "organizationName": organization_name,
                "dataCenter": data_center,
            },
            headers=_auth_headers(origin),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 409:
            raise AuthFlowError(
                "An account with this email already exists. Run `deepeval login` "
                "again and choose “Log in to an existing account”."
            )
        if resp.status_code == 400:
            print(f"❌ {_describe_error(resp, 'Sign up failed')}")
            continue
        if not resp.ok:
            raise AuthFlowError(_describe_error(resp, "Sign up failed"))

        token = resp.json().get("token")
        break

    if not token:
        raise AuthFlowError("Sign up did not return a session token.")

    projects = _list_projects(session, base_url, origin, token)
    project = _select_project(projects)
    return _create_api_key(session, base_url, origin, token, project["id"])


# --------------------------------------------------------------------------- #
# Browser pairing (Google / SSO / signup)
# --------------------------------------------------------------------------- #


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _generate_pairing_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def browser_pairing_login(
    prod_url: str,
    with_utm: Callable[..., str],
    intent: Optional[str] = None,
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

    pair_path = f"{prod_url}/pair?code={pairing_code}&port={port}"
    if intent:
        pair_path += f"&intent={intent}"
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
    print("Waiting for you to select a project in the browser...")

    if result.event.wait(timeout=PAIRING_TIMEOUT_SECONDS):
        return result.api_key

    print("\n⌛ Timed out waiting for the browser.")
    return None
