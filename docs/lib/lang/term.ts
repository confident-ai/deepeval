/**
 * Inline prose terms: the language-agnostic half of `<Term>`. Server-safe (no
 * React) so `lib/source.ts` can share the tag grammar with the component.
 */

import type { Language } from "./languages";

export type TermProps = { py: string; ts: string };

export const TERM_PROPS = {
  python: "py",
  typescript: "ts",
} as const satisfies Record<Language, keyof TermProps>;

const TERM_TAG = /<Term\s([^>]*?)\/>/g;
const ATTRIBUTE = /([a-zA-Z]+)="([^"]*)"/g;

/** mdast emits attribute quotes as character references: `py=&#x22;…&#x22;`. */
const QUOTE_ENTITY = /&(?:#x22|#34|quot);/gi;

/**
 * Rewrite `<Term .../>` to its Python spelling in backticks for the markdown
 * and `/llms.*` surfaces, which serialize mdast without rendering components.
 */
export function lowerTerms(markdown: string): string {
  return markdown.replace(TERM_TAG, (match, attributes: string) => {
    const python = [
      ...attributes.replace(QUOTE_ENTITY, '"').matchAll(ATTRIBUTE),
    ].find(([, name]) => name === TERM_PROPS.python)?.[2];

    if (python === undefined) {
      throw new Error(
        `[lang-term] <Term> is missing a "${TERM_PROPS.python}" spelling: ${match}`,
      );
    }
    return `\`${python}\``;
  });
}
