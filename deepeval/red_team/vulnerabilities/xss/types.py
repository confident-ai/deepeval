from enum import Enum


class XSSType(Enum):
    """
    Enum for Cross-Site Scripting (XSS) vulnerability types.

    - Reflected XSS: user-supplied markup is echoed straight back into the
      response and executes when that response is rendered.
    - Stored XSS: malicious markup is persisted (profile, note, message) and
      later served to other users where it executes.
    - DOM-based XSS: output is written into a client-side DOM sink
      (innerHTML, location, javascript: URI) where it executes.
    """

    REFLECTED_XSS = "reflected_xss"
    STORED_XSS = "stored_xss"
    DOM_BASED_XSS = "dom_based_xss"


# List of all available types for easy access
XSS_TYPES = [
    XSSType.REFLECTED_XSS,
    XSSType.STORED_XSS,
    XSSType.DOM_BASED_XSS,
]
