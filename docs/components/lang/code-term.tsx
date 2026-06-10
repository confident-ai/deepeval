"use client";

import { useLanguage } from "@/components/lang/language-provider";
import { getTerm, type TermId } from "@/lib/lang/terms";

/**
 * Inline language-aware code term. Renders the active language's spelling
 * of `id` (defaulting to Python via `LanguageProvider`) inside a `<code>`.
 *
 * Usage in MDX:  An <C id="class::LLMTestCase"/> has an <C id="field::actual_output"/>.
 *
 * `id` is typed as `TermId` for editor autocomplete, but the real guard is
 * the `throw` inside `getTerm`: an unknown id fails during static
 * generation, surfacing as a hard build error instead of a silent drop.
 */
export function C({ id }: { id: TermId }) {
  const { language } = useLanguage();
  return <code>{getTerm(id, language)}</code>;
}
