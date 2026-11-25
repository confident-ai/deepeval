from enum import Enum


class ShellInjectionType(Enum):
    """
    Enum for Shell Injection vulnerability types.

    - Command injection through unauthorized system command execution
    - System command execution with malicious payloads
    - Shell escape sequences and command manipulation
    """

    COMMAND_INJECTION = "command_injection"
    SYSTEM_COMMAND_EXECUTION = "system_command_execution"
    SHELL_ESCAPE_SEQUENCES = "shell_escape_sequences"


# List of all available types for easy access
SHELL_INJECTION_TYPES = [
    ShellInjectionType.COMMAND_INJECTION,
    ShellInjectionType.SYSTEM_COMMAND_EXECUTION,
    ShellInjectionType.SHELL_ESCAPE_SEQUENCES,
]
