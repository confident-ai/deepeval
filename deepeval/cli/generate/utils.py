import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import typer

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden, ConversationalGolden
from deepeval.synthesizer.config import (
    ConversationalStylingConfig,
    StylingConfig,
)


class GenerationMethod(str, Enum):
    DOCS = "docs"
    CONTEXTS = "contexts"
    SCRATCH = "scratch"
    GOLDENS = "goldens"


class GoldenVariation(str, Enum):
    SINGLE_TURN = "single-turn"
    MULTI_TURN = "multi-turn"


class FileType(str, Enum):
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"


def require_method_option(value, option_name: str, method: GenerationMethod):
    if value is None or value == []:
        raise typer.BadParameter(
            f"`{option_name}` is required when --method is `{method.value}`.",
            param_hint=option_name,
        )
    return value


def load_contexts_file(contexts_file: Path) -> List[List[str]]:
    try:
        raw_contexts = json.loads(contexts_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(
            f"Contexts file not found: {contexts_file}",
            param_hint="--contexts-file",
        )
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Contexts file must be valid JSON: {exc}",
            param_hint="--contexts-file",
        )

    if not isinstance(raw_contexts, list):
        raise typer.BadParameter(
            "Contexts file must contain a JSON list of context lists.",
            param_hint="--contexts-file",
        )

    for context in raw_contexts:
        if not isinstance(context, list) or not all(
            isinstance(chunk, str) for chunk in context
        ):
            raise typer.BadParameter(
                'Contexts file must be shaped like [["chunk 1", "chunk 2"], ...].',
                param_hint="--contexts-file",
            )

    return raw_contexts


def load_goldens_file(
    goldens_file: Path,
) -> Union[List[Golden], List[ConversationalGolden]]:
    if not goldens_file.exists():
        raise typer.BadParameter(
            f"Goldens file not found: {goldens_file}",
            param_hint="--goldens-file",
        )

    dataset = EvaluationDataset()
    suffix = goldens_file.suffix.lower()
    if suffix == ".json":
        dataset.add_goldens_from_json_file(str(goldens_file))
        return dataset.goldens

    if suffix == ".csv":
        dataset.add_goldens_from_csv_file(str(goldens_file))
        return dataset.goldens

    if suffix == ".jsonl":
        dataset.add_goldens_from_jsonl_file(str(goldens_file))
        return dataset.goldens

    raise typer.BadParameter(
        "Goldens file must be a .json, .csv, or .jsonl file.",
        param_hint="--goldens-file",
    )


def validate_golden_variation(
    goldens: Union[List[Golden], List[ConversationalGolden]],
    variation: GoldenVariation,
) -> None:
    if not goldens:
        raise typer.BadParameter(
            "Goldens file does not contain any goldens.",
            param_hint="--goldens-file",
        )

    first_golden = goldens[0]
    is_multi_turn = isinstance(first_golden, ConversationalGolden)
    if variation == GoldenVariation.MULTI_TURN and not is_multi_turn:
        raise typer.BadParameter(
            "`--variation multi-turn` requires conversational goldens.",
            param_hint="--variation",
        )
    if variation == GoldenVariation.SINGLE_TURN and is_multi_turn:
        raise typer.BadParameter(
            "`--variation single-turn` requires single-turn goldens.",
            param_hint="--variation",
        )


def single_turn_styling_config(
    scenario: Optional[str],
    task: Optional[str],
    input_format: Optional[str],
    expected_output_format: Optional[str],
) -> Optional[StylingConfig]:
    if not any([scenario, task, input_format, expected_output_format]):
        return None
    return StylingConfig(
        scenario=scenario,
        task=task,
        input_format=input_format,
        expected_output_format=expected_output_format,
    )


def multi_turn_styling_config(
    scenario_context: Optional[str],
    conversational_task: Optional[str],
    participant_roles: Optional[str],
    scenario_format: Optional[str],
    expected_outcome_format: Optional[str],
) -> Optional[ConversationalStylingConfig]:
    if not any(
        [
            scenario_context,
            conversational_task,
            participant_roles,
            scenario_format,
            expected_outcome_format,
        ]
    ):
        return None
    return ConversationalStylingConfig(
        scenario_context=scenario_context,
        conversational_task=conversational_task,
        participant_roles=participant_roles,
        scenario_format=scenario_format,
        expected_outcome_format=expected_outcome_format,
    )


def validate_scratch_styling(
    variation: GoldenVariation,
    scenario: Optional[str],
    task: Optional[str],
    input_format: Optional[str],
    scenario_context: Optional[str],
    conversational_task: Optional[str],
    participant_roles: Optional[str],
) -> None:
    if variation == GoldenVariation.SINGLE_TURN:
        missing = [
            option
            for option, value in [
                ("--scenario", scenario),
                ("--task", task),
                ("--input-format", input_format),
            ]
            if value is None
        ]
    else:
        missing = [
            option
            for option, value in [
                ("--scenario-context", scenario_context),
                ("--conversational-task", conversational_task),
                ("--participant-roles", participant_roles),
            ]
            if value is None
        ]

    if missing:
        raise typer.BadParameter(
            "Scratch generation requires: " + ", ".join(missing),
            param_hint=missing[0],
        )
