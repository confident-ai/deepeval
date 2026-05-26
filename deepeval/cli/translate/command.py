from __future__ import annotations

from typing import List, Optional, Tuple

import typer
from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from deepeval.cli.translate.translate import (
    find_missing_placeholders,
    load_localized_templates,
    run_translation_llm,
    save_localized_templates,
)
from deepeval.metric_templates.community.languages import (
    is_english,
    parse_language_slug,
    require_valid_language,
)
from deepeval.metric_templates.resolver import (
    MetricTemplateNotFoundError,
    clear_metric_template_cache,
    iter_base_template_methods,
    resolve_base_template,
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


def _resolve_language_slug(lang: str) -> str:
    if is_english(lang):
        print(
            "[red]Error:[/red] English is the default language. Unset [bold]DEEPEVAL_METRIC_TEMPLATE_LANGUAGE[/bold] "
            "instead of explicitly translating to English."
        )
        raise typer.Exit(code=1)
    try:
        return parse_language_slug(lang)
    except ValueError:
        print(f"[yellow]Could not parse language slug[/yellow] from {lang!r}.")
        default_slug = lang.strip().lower().replace("-", "_").replace(" ", "_")
        entered = typer.prompt(
            "Enter a clean language slug (lowercase letters, digits, underscores)",
            default=default_slug,
        ).strip()
        try:
            return parse_language_slug(entered)
        except ValueError as e:
            print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e


def translate_command(
    lang: str = typer.Argument(
        ...,
        metavar="LANG",
        help="Target language slug (e.g. hindi, vietnamese).",
    ),
    metrics: Annotated[
        Optional[List[str]],
        typer.Option(
            "--metrics",
            help=(
                "The name of the metric class (e.g., FaithfulnessMetric). "
                "Repeat the flag or use a comma-separated list."
            ),
            show_default=False,
        ),
    ] = None,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Optional model name or instance path; uses the same default evaluation model if unset.",
    ),
    contribute: bool = typer.Option(
        False,
        "--contribute",
        help=(
            "Write into metric_templates/community/ for upstream OSS PRs. "
            "LANG must already exist in the MetricTemplateLanguage Enum."
        ),
    ),
) -> None:
    """Translate shipped English metric templates into LANG via your configured LLM."""
    if not metrics:
        print("[red]Error:[/red] Pass at least one [bold]--metrics[/bold] class name.")
        raise typer.Exit(code=1)

    slug = _resolve_language_slug(lang)
    if contribute:
        try:
            slug = require_valid_language(slug)
        except ValueError as e:
            print(f"\n[red]Contribution Error:[/red] {e}")
            raise typer.Exit(code=1) from e

    metrics = _normalize_metric_names(metrics)
    metric_jobs: List[Tuple[str, List[Tuple[str, str]]]] = []
    
    for class_name in metrics:
        if class_name in _RESERVED_TRANSLATE_CLASSES:
            print(f"[yellow]Skipping reserved class[/yellow] {class_name!r} (not a valid metric template).")
            continue
        try:
            pairs = iter_base_template_methods(class_name)
        except MetricTemplateNotFoundError as e:
            print(f"[red]{class_name}:[/red] {e}")
            raise typer.Exit(code=1)
        metric_jobs.append((class_name, pairs))

    existing = load_localized_templates(slug, contribute=contribute)

    # Simplified diffing logic
    jobs_to_translate: List[Tuple[str, str, str]] = []
    skipped = 0
    
    for class_name, pairs in metric_jobs:
        existing_methods = existing.get(class_name, {})
        for method, source in pairs:
            # If the method exists and has content, skip it
            if existing_methods.get(method):
                skipped += 1
            else:
                jobs_to_translate.append((class_name, method, source))

    if not jobs_to_translate:
        if skipped:
            print(f"[yellow]Nothing to translate[/yellow] ({skipped} template(s) already exist for {slug}).")
        else:
            print("[yellow]Nothing to translate[/yellow] (no template methods found).")
        raise typer.Exit(code=0)

    updates: dict[str, dict[str, str]] = {}
    
    with Progress(
        SpinnerColumn(style="rgb(106,0,255)"),
        BarColumn(bar_width=60),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("", total=len(jobs_to_translate))
        
        for class_name, method, source in jobs_to_translate:
            progress.update(
                task_id,
                description=(f"Translating [rgb(106,0,255)]'{class_name}.{method}'[/rgb(106,0,255)]…"),
            )
            
            try:
                prompt = resolve_base_template(
                    "TranslateCLI",
                    "prompt",
                    lang=slug,
                    metric_class_name=class_name,
                    metric_method_name=method,
                    source=source,
                )
            except Exception as e:
                print(f"[red]Failed to render translation prompt[/red] for {class_name!r}.{method!r}: {e}")
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
                    f"these Jinja tokens were lost during translation: {missing}"
                )

            updates.setdefault(class_name, {})[method] = rewritten
            progress.advance(task_id, 1)

    if not updates:
        print("[yellow]Nothing to write[/yellow] (no metric classes processed).")
        raise typer.Exit(code=0)

    out_path = save_localized_templates(slug, updates, contribute=contribute)
    clear_metric_template_cache()
    
    n_metrics = len(updates)
    n_templates = sum(len(v) for v in updates.values())
    metric_word = "metric" if n_metrics == 1 else "metrics"
    template_word = "template" if n_templates == 1 else "templates"
    skip_note = f", skipped {skipped} existing" if skipped else ""
    
    print(
        f"[green]✅ Translated {n_metrics} {metric_word} "
        f"({n_templates} {template_word}) to {slug}{skip_note}[/green]"
    )
    
    # Context-aware success message
    if contribute:
        print(
            "\n[green]Translation complete![/green] \n"
            f"Please commit your new JSON file ([bold]{out_path.name}[/bold]) and open a Pull Request\n"
            "on the DeepEval GitHub repository to share this language with the community ❤️"
        )
    else:
        print(
            f"\n[green]Translation saved to {out_path}[/green]\n"
            f"Set [bold]DEEPEVAL_METRIC_TEMPLATE_LANGUAGE={slug}[/bold] in your environment to use these templates."
        )
