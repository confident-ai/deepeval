from typing import List, Literal, Optional, Union
from dataclasses import dataclass, field

from deepeval.test_run.test_run import TestRunResultDisplay


@dataclass
class AsyncConfig:
    run_async: bool = True
    throttle_value: float = 0
    max_concurrent: int = 20

    def __post_init__(self):
        if self.max_concurrent < 1:
            raise ValueError("'max_concurrent' must be at least 1")
        if self.throttle_value < 0:
            raise ValueError("'throttle_value' must be at least 0")


@dataclass
class DisplayConfig:
    show_indicator: bool = True
    print_results: bool = True
    verbose_mode: Optional[bool] = None
    display_option: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL
    # Optional folder for timestamped `test_run_<ts>.json` exports.
    results_folder: Optional[str] = None
    results_subfolder: Optional[str] = None
    truncate_passing_cases: bool = True
    # Offer `deepeval inspect` after the run. Honors `DEEPEVAL_NO_INSPECT_PROMPT=1`.
    inspect_after_run: bool = True
    # Deprecated: writes one .log per TestResult. Prefer `results_folder`, which
    # saves the full TestRun as a structured JSON readable by AI tools.
    file_type: Literal["html", "md"] = None
    file_output_dir: Optional[str] = None
    # Optional grouping for the aggregate metrics table.  Pass a tag string to
    # slice by that tag (cases that carry the tag belong to the group), or a
    # metadata key to slice by the value stored under that key.  The overall
    # pooled row is always printed first; per-group rows follow.
    # Examples:
    #   group_by="short-answer"          (a tag)
    #   group_by="difficulty"            (a metadata key)
    #   group_by=["short-answer", "multi-hop"]  (several tags, one row each)
    group_by: Optional[Union[str, List[str]]] = None


@dataclass
class CacheConfig:
    write_cache: bool = True
    use_cache: bool = False


@dataclass
class ErrorConfig:
    ignore_errors: bool = False
    skip_on_missing_params: bool = False
