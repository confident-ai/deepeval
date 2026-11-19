import sys
from typing import List

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import ToolCall
from deepeval.utils import serialize_to_json


def print_tools_called(tools_called_list: List[ToolCall]):
    if not tools_called_list:
        return ""
    string = "[\n"
    for index, tools_called in enumerate(tools_called_list):
        json_string = serialize_to_json(tools_called, indent=4)
        indented_json_string = "\n".join(
            "  " + line for line in json_string.splitlines()
        )
        string += indented_json_string
        if index < len(tools_called_list) - 1:
            string += ",\n"
        else:
            string += "\n"
    string += "]"
    return string


def print_verbose_logs(metric: str, logs: str):
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write(f"{metric} Verbose Logs\n")
    sys.stdout.write("*" * 50 + "\n")
    sys.stdout.write("\n")
    sys.stdout.write(logs + "\n")
    sys.stdout.write("\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.flush()


def construct_verbose_logs(metric: BaseMetric, steps: List[str]) -> str:
    verbose_logs = ""
    for i in range(len(steps) - 1):
        verbose_logs += steps[i]

        # don't add new line for penultimate step
        if i < len(steps) - 2:
            verbose_logs += " \n \n"
    dropped = getattr(metric, "qag_verdicts_dropped_count", 0)
    if dropped:
        # Surface silently excluded judge verdicts (see issue #3654): the
        # count is also on the metric for programmatic inspection.
        received = getattr(metric, "qag_verdicts_received_count", 0)
        if verbose_logs:
            verbose_logs += " \n \n"
        verbose_logs += (
            f"QAG verdict parsing: {dropped} of {received} returned "
            "verdicts were unrecognized or unsupported and excluded."
        )
    if metric.verbose_mode:
        # only print reason and score for deepeval
        print_verbose_logs(metric.__name__, verbose_logs + f"\n \n{steps[-1]}")

    return verbose_logs
