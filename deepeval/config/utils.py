import os
from typing import Any, Optional

_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on", "enable", "enabled"})
_FALSY = frozenset({"0", "false", "f", "no", "n", "off", "disable", "disabled"})


def parse_bool(value: Any, default: bool = False) -> bool:
    """
    Parse an arbitrary value into a boolean using env style semantics.

    Truthy tokens (case-insensitive, quotes/whitespace ignored):
      1, true, t, yes, y, on, enable, enabled
    Falsy tokens:
      0, false, f, no, n, off, disable, disabled

    - bool -> returned as is
    - None -> returns `default`
    - int/float -> False if == 0, else True
    - str/other -> matched against tokens above; non-matching -> `default`

    Args:
        value: Value to interpret.
        default: Value to return if `value` is None or doesn’t match any token.

    Returns:
        The interpreted boolean.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0

    s = str(value).strip().strip('"').strip("'").lower()
    if not s:
        return default
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Read an environment variable and parse it as a boolean using `parse_bool`.

    Args:
        key: Environment variable name.
        default: Returned when the variable is unset or does not match any token.

    Returns:
        Parsed boolean value.
    """
    return parse_bool(os.getenv(key), default)


def bool_to_env_str(value: bool) -> str:
    """
    Canonicalize a boolean to the env/dotenv string form: "1" or "0".

    Args:
        value: Boolean to serialize.

    Returns:
        "1" if True, "0" if False.
    """
    return "1" if bool(value) else "0"


def set_env_bool(key: str, value: Optional[bool] = False) -> None:
    """
    Set an environment variable to a canonical boolean string ("1" or "0").

    Args:
        key: The environment variable name to set.
        value: The boolean value to store. If None, it is treated as False.
               True -> "1", False/None -> "0".

    Notes:
        - This function always overwrites the variable in `os.environ`.
        - Use `get_env_bool` to read back and parse the value safely.
    """
    os.environ[key] = bool_to_env_str(bool(value))
