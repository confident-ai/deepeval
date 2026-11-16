import fs from "node:fs";
import path from "node:path";

/**
 * Blog posts are folders: `content/blog/<yyyy_mmm>/<slug>/article.mdx`, with an
 * optional cover and frame data beside the article. The URL is `/blog/<slug>`.
 */
export const BLOG_DIR = path.join(process.cwd(), "content/blog");

export const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
export const MONTH_FOLDER = new RegExp(`^\\d{4}_(${MONTHS.join("|")})$`);
export const SLUG = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const POST_FILE = /^(\d{4}_[a-z]{3})\/([^/]+)\/article\.mdx$/;
export const COVER_FILE = /^cover\.(webp|png|jpe?g|gif)$/;

/** Slugs for a blog file path relative to `content/blog`; `undefined` keeps fumadocs' default. */
export function postSlugs(file) {
  const match = POST_FILE.exec(file.path);
  return match ? [match[2]] : undefined;
}

/** The month folder a post dated `date` belongs in, e.g. `2026_jul`. */
export function monthFolder(date) {
  const d = new Date(date);
  return `${d.getUTCFullYear()}_${MONTHS[d.getUTCMonth()]}`;
}

/** The public URL of a post's cover, given its path relative to `content/blog`. */
export function postCover(postPath) {
  const dir = path.join(BLOG_DIR, path.dirname(postPath));
  if (!fs.existsSync(dir)) return undefined;
  const file = fs.readdirSync(dir).find((f) => COVER_FILE.test(f));
  return file ? `/media/covers/${path.basename(dir)}${path.extname(file)}` : undefined;
}
