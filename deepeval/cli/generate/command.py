from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.cli.generate.utils import (
    FileType,
    GenerationMethod,
    GoldenVariation,
    load_contexts_file,
    load_goldens_file,
    multi_turn_styling_config,
    require_method_option,
    single_turn_styling_config,
    validate_golden_variation,
    validate_scratch_styling,
)


def generate_command(
    method: GenerationMethod = typer.Option(
        ...,
        "--method",
        help="Golden generation method to use.",
        case_sensitive=False,
    ),
    variation: GoldenVariation = typer.Option(
        ...,
        "--variation",
        help="Golden variation to generate.",
        case_sensitive=False,
    ),
    output_dir: str = typer.Option(
        "./synthetic_data",
        "--output-dir",
        help="Directory where generated goldens will be saved.",
    ),
    file_type: FileType = typer.Option(
        FileType.JSON,
        "--file-type",
        help="File type to save generated goldens as.",
        case_sensitive=False,
    ),
    file_name: Optional[str] = typer.Option(
        None,
        "--file-name",
        help="Optional output filename without extension.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model to use for generation.",
    ),
    async_mode: bool = typer.Option(
        True,
        "--async-mode/--sync-mode",
        help="Whether to generate goldens concurrently.",
    ),
    max_concurrent: int = typer.Option(
        100,
        "--max-concurrent",
        help="Maximum number of concurrent generation tasks.",
    ),
    include_expected: bool = typer.Option(
        True,
        "--include-expected/--no-include-expected",
        help="Whether to generate expected output or expected outcome.",
    ),
    cost_tracking: bool = typer.Option(
        False,
        "--cost-tracking",
        help="Print generation cost when supported by the model.",
    ),
    documents: Optional[List[str]] = typer.Option(
        None,
        "--documents",
        help="Document path to use with --method docs. Can be passed multiple times.",
    ),
    contexts_file: Optional[Path] = typer.Option(
        None,
        "--contexts-file",
        help='JSON file shaped like [["chunk 1", "chunk 2"], ...].',
    ),
    goldens_file: Optional[Path] = typer.Option(
        None,
        "--goldens-file",
        help="Existing goldens file to augment (.json, .csv, or .jsonl).",
    ),
    num_goldens: Optional[int] = typer.Option(
        None,
        "--num-goldens",
        help="Number of goldens to generate with --method scratch.",
    ),
    max_goldens_per_context: int = typer.Option(
        2,
        "--max-goldens-per-context",
        help="Maximum goldens to generate per context.",
    ),
    max_goldens_per_golden: int = typer.Option(
        2,
        "--max-goldens-per-golden",
        help="Maximum goldens to generate per existing golden.",
    ),
    max_contexts_per_document: int = typer.Option(
        3,
        "--max-contexts-per-document",
        help="Maximum contexts to construct per document.",
    ),
    min_contexts_per_document: int = typer.Option(
        1,
        "--min-contexts-per-document",
        help="Minimum contexts to construct per document.",
    ),
    chunk_size: int = typer.Option(
        1024,
        "--chunk-size",
        help="Token chunk size for document parsing.",
    ),
    chunk_overlap: int = typer.Option(
        0,
        "--chunk-overlap",
        help="Token overlap between document chunks.",
    ),
    context_quality_threshold: float = typer.Option(
        0.5,
        "--context-quality-threshold",
        help="Minimum context quality threshold.",
    ),
    context_similarity_threshold: float = typer.Option(
        0.0,
        "--context-similarity-threshold",
        help="Minimum context grouping similarity threshold.",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum retries for context construction quality checks.",
    ),
    scenario: Optional[str] = typer.Option(
        None,
        "--scenario",
        help="Single-turn generation scenario.",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Single-turn generation task.",
    ),
    input_format: Optional[str] = typer.Option(
        None,
        "--input-format",
        help="Single-turn input format.",
    ),
    expected_output_format: Optional[str] = typer.Option(
        None,
        "--expected-output-format",
        help="Single-turn expected output format.",
    ),
    scenario_context: Optional[str] = typer.Option(
        None,
        "--scenario-context",
        help="Multi-turn scenario context.",
    ),
    conversational_task: Optional[str] = typer.Option(
        None,
        "--conversational-task",
        help="Multi-turn conversational task.",
    ),
    participant_roles: Optional[str] = typer.Option(
        None,
        "--participant-roles",
        help="Multi-turn participant roles.",
    ),
    scenario_format: Optional[str] = typer.Option(
        None,
        "--scenario-format",
        help="Multi-turn scenario format.",
    ),
    expected_outcome_format: Optional[str] = typer.Option(
        None,
        "--expected-outcome-format",
        help="Multi-turn expected outcome format.",
    ),
):
    """Generate synthetic goldens with the golden synthesizer."""
    document_paths = None
    contexts = None
    goldens = None

    if method == GenerationMethod.DOCS:
        document_paths = require_method_option(documents, "--documents", method)
    elif method == GenerationMethod.CONTEXTS:
        contexts_path = require_method_option(
            contexts_file, "--contexts-file", method
        )
        contexts = load_contexts_file(contexts_path)
    elif method == GenerationMethod.SCRATCH:
        require_method_option(num_goldens, "--num-goldens", method)
        validate_scratch_styling(
            variation=variation,
            scenario=scenario,
            task=task,
            input_format=input_format,
            scenario_context=scenario_context,
            conversational_task=conversational_task,
            participant_roles=participant_roles,
        )
    elif method == GenerationMethod.GOLDENS:
        goldens_path = require_method_option(
            goldens_file, "--goldens-file", method
        )
        goldens = load_goldens_file(goldens_path)
        validate_golden_variation(goldens, variation)

    styling_config = single_turn_styling_config(
        scenario=scenario,
        task=task,
        input_format=input_format,
        expected_output_format=expected_output_format,
    )
    conversational_styling_config = multi_turn_styling_config(
        scenario_context=scenario_context,
        conversational_task=conversational_task,
        participant_roles=participant_roles,
        scenario_format=scenario_format,
        expected_outcome_format=expected_outcome_format,
    )
    synthesizer = Synthesizer(
        model=model,
        async_mode=async_mode,
        max_concurrent=max_concurrent,
        styling_config=styling_config,
        conversational_styling_config=conversational_styling_config,
        cost_tracking=cost_tracking,
    )

    if method == GenerationMethod.DOCS:
        context_construction_config = ContextConstructionConfig(
            max_contexts_per_document=max_contexts_per_document,
            min_contexts_per_document=min_contexts_per_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_quality_threshold=context_quality_threshold,
            context_similarity_threshold=context_similarity_threshold,
            max_retries=max_retries,
        )
        if variation == GoldenVariation.SINGLE_TURN:
            synthesizer.generate_goldens_from_docs(
                document_paths=document_paths,
                include_expected_output=include_expected,
                max_goldens_per_context=max_goldens_per_context,
                context_construction_config=context_construction_config,
            )
        else:
            synthesizer.generate_conversational_goldens_from_docs(
                document_paths=document_paths,
                include_expected_outcome=include_expected,
                max_goldens_per_context=max_goldens_per_context,
                context_construction_config=context_construction_config,
            )

    elif method == GenerationMethod.CONTEXTS:
        if variation == GoldenVariation.SINGLE_TURN:
            synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=include_expected,
                max_goldens_per_context=max_goldens_per_context,
            )
        else:
            synthesizer.generate_conversational_goldens_from_contexts(
                contexts=contexts,
                include_expected_outcome=include_expected,
                max_goldens_per_context=max_goldens_per_context,
            )

    elif method == GenerationMethod.SCRATCH:
        if variation == GoldenVariation.SINGLE_TURN:
            synthesizer.generate_goldens_from_scratch(num_goldens=num_goldens)
        else:
            synthesizer.generate_conversational_goldens_from_scratch(
                num_goldens=num_goldens
            )

    elif method == GenerationMethod.GOLDENS:
        if variation == GoldenVariation.SINGLE_TURN:
            synthesizer.generate_goldens_from_goldens(
                goldens=goldens,
                max_goldens_per_golden=max_goldens_per_golden,
                include_expected_output=include_expected,
            )
        else:
            synthesizer.generate_conversational_goldens_from_goldens(
                goldens=goldens,
                max_goldens_per_golden=max_goldens_per_golden,
                include_expected_outcome=include_expected,
            )

    output_path = synthesizer.save_as(
        file_type=file_type.value,
        directory=output_dir,
        file_name=file_name,
        quiet=True,
    )
    print(f"Synthetic goldens saved at {output_path}!")
