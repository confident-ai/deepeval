from typing import Optional
from dataclasses import dataclass

from deepeval.test_run.test_run import TestRunResultDisplay


@dataclass
class AsyncConfig:
    run_async: bool = True
    throttle_value: int = 0
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
    file_output_dir: Optional[str] = None


@dataclass
class CacheConfig:
    write_cache: bool = True
    use_cache: bool = False


@dataclass
class ErrorConfig:
    ignore_errors: bool = False
    skip_on_missing_params: bool = False
