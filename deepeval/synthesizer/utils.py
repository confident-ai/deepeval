from enum import Enum
from typing import Optional

# ANSI escape codes for coloring
RESET = "\033[0m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"


class SynthesizerStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


def print_synthesizer_status(
    trace_worker_status: SynthesizerStatus,
    message: str,
    description: Optional[str] = None,
):
    prefix = f"{DIM}[Confident AI Synthesizer Log]{RESET}"
    if trace_worker_status == SynthesizerStatus.SUCCESS:
        colored_msg = f"{GREEN}{message}{RESET}"
        label = "SUCCESS"
    elif trace_worker_status == SynthesizerStatus.FAILURE:
        colored_msg = f"{RED}{message}{RESET}"
        label = "FAILURE"
    elif trace_worker_status == SynthesizerStatus.WARNING:
        colored_msg = f"{YELLOW}{message}{RESET}"
        label = "WARNING"
    if description:
        print(f"{prefix} {label}: {colored_msg}: {description}")
    else:
        print(f"{prefix} {label}: {colored_msg}")
