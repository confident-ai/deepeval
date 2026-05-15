from __future__ import annotations

from typing import List, Optional, Tuple

import typer
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from deepeval.cli.translate.translate import (
    find_missing_placeholders,
    run_translation_llm,
    save_translated_templates,
)
from deepeval.metric_templates.resolver import (
    MetricTemplateNotFoundError,
    iter_bundle_template_methods,
    resolve_template,
)

_RESERVED_TRANSLATE_CLASSES = frozenset({"TranslateCLI"})


def _normalize_metric_names(metrics: List[str]) -> List[str]:
    """Expand comma-separated entries and strip whitespace (one argv token can be ``A,B``)."""
    out: List[str] = []
    for raw in metrics:
        for part in raw.replace(",", " ").split():
            part = part.strip()
            if part:
                out.append(part)
    return out


def translate_command(
    lang: str = typer.Argument(
        ...,
        metavar="LANG",
        help="Target language (e.g. Spanish, German, Japanese).",
    ),
    metrics: Annotated[
        Optional[List[str]],
        typer.Option(
            "--metrics",
            help=(
                "Bundle metric class name(s), e.g. FaithfulnessMetric. "
                "Repeat the flag or use a comma-separated list in one value."
            ),
            show_default=False,
        ),
    ] = None,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Optional model name or instance path; uses the same defaults as metrics.",
    ),
) -> None:
    """Translate shipped English metric templates into LANG via your configured LLM.

    Writes merged results to ``.deepeval/templates.json``.
    """
    if not metrics:
        print("[red]Error:[/red] pass at least one [bold]--metrics[/bold] class name.")
        raise typer.Exit(code=1)

    metrics = _normalize_metric_names(metrics)

    metric_jobs: List[Tuple[str, List[Tuple[str, str]]]] = []
    for class_name in metrics:
        if class_name in _RESERVED_TRANSLATE_CLASSES:
            print(
                f"[yellow]Skipping reserved class[/yellow] {class_name!r} "
                "(not a metric template root)."
            )
            continue
        try:
            pairs = iter_bundle_template_methods(class_name)
        except MetricTemplateNotFoundError as e:
            print(f"[red]{class_name}:[/red] {e}")
            raise typer.Exit(code=1)
        metric_jobs.append((class_name, pairs))

    total_steps = sum(len(pairs) for _, pairs in metric_jobs)
    if total_steps == 0:
        print("[yellow]Nothing to translate[/yellow] (no template methods found).")
        raise typer.Exit(code=0)

    updates: dict[str, dict[str, str]] = {}
    with Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        BarColumn(bar_width=60),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("", total=total_steps)
        for class_name, pairs in metric_jobs:
            class_out: dict[str, str] = {}
            for method, source in pairs:
                progress.update(
                    task_id,
                    description=(
                        f"Translating [rgb(106,0,255)]'{class_name}'[/rgb(106,0,255)]…"
                    ),
                )
                try:
                    prompt = resolve_template(
                        "TranslateCLI",
                        "prompt",
                        lang=lang,
                        metric_class_name=class_name,
                        metric_method_name=method,
                        source=source,
                    )
                except Exception as e:
                    print(
                        f"[red]Failed to render translation prompt[/red] for "
                        f"{class_name!r}.{method!r}: {e}"
                    )
                    raise typer.Exit(code=1)

                try:
                    rewritten = run_translation_llm(
                        prompt=prompt,
                        model=model,
                    )
                except Exception as exc:
                    print(f"[red]LLM error[/red] for {class_name!r}.{method!r}: {exc}")
                    raise typer.Exit(code=1) from exc

                missing = find_missing_placeholders(source, rewritten)
                if missing:
                    print(
                        f"[yellow]Warning:[/yellow] {class_name!r}.{method!r} — "
                        f"these Jinja tokens from the source were not found "
                        f"verbatim in the output: {missing}"
                    )

                class_out[method] = rewritten
                progress.advance(task_id, 1)

            if class_out:
                updates[class_name] = class_out

    if not updates:
        print("[yellow]Nothing to write[/yellow] (no metric classes processed).")
        raise typer.Exit(code=0)

    save_translated_templates(updates)
    n_metrics = len(updates)
    n_templates = sum(len(v) for v in updates.values())
    metric_word = "metric" if n_metrics == 1 else "metrics"
    template_word = "template" if n_templates == 1 else "templates"
    print(
        f"[green]✅ Translated {n_metrics} {metric_word} "
        f"({n_templates} {template_word}) to {lang}[/green]"
    )