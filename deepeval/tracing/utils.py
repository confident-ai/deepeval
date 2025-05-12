from enum import Enum


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"


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
