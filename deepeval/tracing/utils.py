from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from deepeval.metrics import BaseMetric
from deepeval.tracing.api import MetricData


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
    TESTING = "testing"


def validate_environment(environment: str):
    if environment not in [env.value for env in Environment]:
        valid_values = ", ".join(f'"{env.value}"' for env in Environment)
        raise ValueError(
            f"Invalid environment: {environment}. Please use one of the following instead: {valid_values}"
        )


def validate_sampling_rate(sampling_rate: float):
    if sampling_rate < 0 or sampling_rate > 1:
        raise ValueError(
            f"Invalid sampling rate: {sampling_rate}. Please use a value between 0 and 1"
        )


def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {
            key: make_json_serializable(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")  # optional: exclude private attrs
        }
    return str(obj)


def to_zod_compatible_iso(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def perf_counter_to_datetime(perf_counter_value: float) -> datetime:
    """
    Convert a perf_counter value to a datetime object.

    Args:
        perf_counter_value: A float value from perf_counter()

    Returns:
        A datetime object representing the current time
    """
    # Get the current time
    current_time = datetime.now(timezone.utc)
    # Calculate the time difference in seconds
    time_diff = current_time.timestamp() - perf_counter()
    # Convert perf_counter value to a real timestamp
    timestamp = time_diff + perf_counter_value
    # Return as a datetime object
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

# copied from deepeval.evaluate.utils due to circular import
def create_metric_data(metric: BaseMetric) -> MetricData:
    if metric.error is not None:
        return MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=False,
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )
    else:
        return MetricData(
            name=metric.__name__,
            score=metric.score,
            threshold=metric.threshold,
            reason=metric.reason,
            success=metric.is_successful(),
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=None,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )