/**
 * Single source of truth for blog author metadata.
 *
 * Ported from the old Docusaurus `blog/authors.yml`. Keeping this as a
 * typed TS module (instead of YAML) means:
 *   - Every entry is compile-time checked to have all required fields
 *     (via `satisfies Record<string, Author>`).
 *   - `AuthorId` is a literal union (`"penguine" | "kritinv" | ...`) so
 *     Zod can use `z.enum(AUTHOR_IDS)` to validate frontmatter at build
 *     time — a typo in a post's `authors: [...]` array fails the build
 *     with a path like `content/blog/foo.mdx: authors[0]`.
 *   - `getAuthor(id)` returns a fully-typed `Author` with no casts.
 */

export type Author = {
  readonly name: string;
  readonly title: string;
  readonly url: string;
  readonly imageUrl: string;
};

export const authors = {
  penguine: {
    name: "Jeffrey Ip",
    title: "DeepEval Wizard",
    url: "https://github.com/penguine-ip",
    imageUrl: "https://github.com/penguine-ip.png",
  },
  kritinv: {
    name: "Kritin Vongthongsri",
    title: "DeepEval Guru",
    url: "https://github.com/kritinv",
    imageUrl: "https://github.com/kritinv.png",
  },
  cale: {
    name: "Cale",
    title: "DeepEval Scribe",
    url: "https://github.com/A-Vamshi",
    imageUrl: "https://github.com/A-Vamshi.png",
  },
} as const satisfies Record<string, Author>;

export type AuthorId = keyof typeof authors;

/**
 * Frozen tuple of all known author IDs. Typed as a non-empty tuple so
 * it's directly usable by `z.enum(...)` which requires that shape.
 */
export const AUTHOR_IDS = Object.keys(authors) as [AuthorId, ...AuthorId[]];

export function getAuthor(id: AuthorId): Author {
  return authors[id];
}
