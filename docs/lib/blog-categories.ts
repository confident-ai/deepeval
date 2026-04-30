/**
 * Single source of truth for blog categories.
 *
 * Intentionally mirrors the section headings in `content/blog/meta.json`
 * (`---[Icon]Label---`) so the per-post `category` frontmatter lines up
 * 1:1 with the sidebar groupings — one place to rename or add to.
 *
 * Shape + conventions follow `lib/authors.ts`:
 *   - `BlogCategory` is the value type (label + Lucide icon name).
 *   - `blogCategories` is a frozen `satisfies` record so each entry is
 *     compile-time checked.
 *   - `BlogCategoryId` is a literal union of the keys, used by
 *     `z.enum(BLOG_CATEGORY_IDS)` in `source.config.ts` to validate
 *     frontmatter at build time.
 */
import type { LucideIcon } from "lucide-react";
import { Megaphone, Users, Scale } from "lucide-react";

export type BlogCategory = {
  readonly label: string;
  readonly icon: LucideIcon;
};

export const blogCategories = {
  announcements: { label: "Announcements", icon: Megaphone },
  community: { label: "Community", icon: Users },
  comparisons: { label: "Comparisons", icon: Scale },
} as const satisfies Record<string, BlogCategory>;

export type BlogCategoryId = keyof typeof blogCategories;

export const BLOG_CATEGORY_IDS = Object.keys(blogCategories) as [
  BlogCategoryId,
  ...BlogCategoryId[],
];

export function getBlogCategory(id: BlogCategoryId): BlogCategory {
  return blogCategories[id];
}
