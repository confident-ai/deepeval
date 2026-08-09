import pageLanguages from "../generated/page-languages.json";
import type { Language } from "./languages";

/**
 * URL → frontmatter `languages`, produced by
 * `scripts/generate-page-languages.mjs`. Safe to import from client components
 * (plain JSON — no fumadocs server collections).
 */
const PAGE_LANGUAGES = pageLanguages as Record<string, Language[]>;

/** Frontmatter languages for this pathname, or undefined if undeclared. */
export function getPageLanguages(pathname: string): Language[] | undefined {
  return PAGE_LANGUAGES[pathname];
}
