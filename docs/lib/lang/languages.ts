/**
 * Single source of truth for the languages the docs support.
 *
 * Everything derives from `LANGUAGES`: the `Language` type, the
 * `LANGUAGE_IDS` tuple that `z.enum` uses to validate the `languages`
 * frontmatter field in `source.config.ts`, and the language selector's
 * options. Adding an entry here is the only edit needed to make a language
 * declarable and selectable.
 *
 * Shape + conventions follow `lib/blog-categories.ts`.
 */

export type LanguageMeta = {
  readonly label: string;
  readonly icon: string;
  readonly description: string;
  readonly tag?: string;
};

export const LANGUAGES = {
  python: {
    label: "Python",
    icon: "/icons/python.svg",
    description: "First class support",
  },
  typescript: {
    label: "TypeScript",
    icon: "/icons/typescript.svg",
    description: "Beta release",
    tag: "Beta",
  },
} as const satisfies Record<string, LanguageMeta>;

export type Language = keyof typeof LANGUAGES;

export const LANGUAGE_IDS = Object.keys(LANGUAGES) as [
  Language,
  ...Language[],
];

/** Explicit rather than `LANGUAGE_IDS[0]` so reordering cannot change it. */
export const DEFAULT_LANGUAGE: Language = "python";

/**
 * Language to open a page in when the reader has no usable preference yet.
 *
 * Mono-language pages use their only language (so TS-only pages SSR as
 * TypeScript instead of a Python 501). Bilingual / undeclared pages keep the
 * product default (Python).
 */
export function resolveInitialLanguage(supported?: Language[]): Language {
  if (supported?.length === 1) return supported[0];
  return DEFAULT_LANGUAGE;
}
