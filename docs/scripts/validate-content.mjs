#!/usr/bin/env node
/**
 * Content checks that the MDX compiler can't make: blog post folders, the
 * media library, and that no content image is loaded from a remote host.
 */
import fs from "node:fs";
import path from "node:path";
import { BLOG_DIR, COVER_FILE, FRAME_FILES, MONTH_FOLDER, SLUG, monthFolder } from "../lib/post-folders.mjs";
import { IMAGE_FILE, IMAGE_FOLDERS, INDEX_FILE, MEDIA_DIR, NAME, indexFolders, loadLibrary } from "../lib/media/library.mjs";

const ROOT = process.cwd();
const errors = [];
const fail = (file, message) => errors.push(`${path.relative(ROOT, file)}: ${message}`);

const walk = (dir, keep) =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p, keep) : keep(p) ? [p] : [];
  });

// ---- Blog post folders ----

const posts = [];
for (const entry of fs.readdirSync(BLOG_DIR, { withFileTypes: true })) {
  const month = path.join(BLOG_DIR, entry.name);
  if (!entry.isDirectory()) {
    if (!["index.mdx", "meta.json"].includes(entry.name)) fail(month, "posts go in <yyyy_mmm>/<slug>/article.mdx");
    continue;
  }
  if (!MONTH_FOLDER.test(entry.name)) {
    fail(month, "month folders are named <yyyy_mmm>, e.g. 2026_jul");
    continue;
  }
  for (const post of fs.readdirSync(month, { withFileTypes: true })) {
    const dir = path.join(month, post.name);
    if (!post.isDirectory()) {
      fail(dir, "a month folder holds only post folders");
      continue;
    }
    if (!SLUG.test(post.name)) fail(dir, "post folder names are kebab-case slugs");
    const files = fs.readdirSync(dir, { withFileTypes: true });
    if (!files.some((f) => f.name === "article.mdx")) fail(dir, "missing article.mdx");
    if (files.filter((f) => COVER_FILE.test(f.name)).length > 1) fail(dir, "a post has at most one cover");
    for (const f of files) {
      const ok = f.isFile() && (f.name === "article.mdx" || COVER_FILE.test(f.name) || FRAME_FILES.includes(f.name));
      if (!ok) fail(path.join(dir, f.name), `a post folder holds only article.mdx, one cover.* and ${FRAME_FILES.join(", ")}`);
    }
    posts.push({ month: entry.name, slug: post.name, file: path.join(dir, "article.mdx") });
  }
}

const seen = new Map();
for (const post of posts) {
  if (seen.has(post.slug)) fail(post.file, `slug "${post.slug}" is also used by ${path.relative(ROOT, seen.get(post.slug))}`);
  seen.set(post.slug, post.file);
  if (!fs.existsSync(post.file)) continue;
  const frontmatter = /^---\n([\s\S]*?)\n---/.exec(fs.readFileSync(post.file, "utf8"))?.[1] ?? "";
  const date = /^date:\s*(.+)$/m.exec(frontmatter)?.[1].trim();
  if (!date) fail(post.file, "missing `date` frontmatter");
  else if (monthFolder(date) !== post.month) fail(post.file, `dated ${date}, so it belongs in ${monthFolder(date)}/`);
  if (/^image:/m.test(frontmatter)) fail(post.file, "`image:` is gone; put a cover.webp beside article.mdx");
}

const meta = JSON.parse(fs.readFileSync(path.join(BLOG_DIR, "meta.json"), "utf8"));
const listed = new Set(meta.pages);
for (const post of posts) {
  if (!listed.has(`${post.month}/${post.slug}/article`)) fail(post.file, `not listed in meta.json as "${post.month}/${post.slug}/article"`);
}
for (const page of meta.pages) {
  if (/^\d{4}_/.test(page) && !fs.existsSync(path.join(BLOG_DIR, `${page}.mdx`))) fail(path.join(BLOG_DIR, "meta.json"), `"${page}" has no post`);
}

// ---- Media library ----

const library = loadLibrary();
for (const folder of indexFolders()) {
  const dir = path.join(MEDIA_DIR, folder);
  if (!IMAGE_FOLDERS.includes(folder)) fail(dir, "not an image folder; add it to IMAGE_FOLDERS in lib/media/library.mjs");
  const listedFiles = new Set();
  for (const image of JSON.parse(fs.readFileSync(path.join(dir, INDEX_FILE), "utf8")).images) {
    listedFiles.add(image.file);
    const name = path.basename(image.file, path.extname(image.file));
    if (!IMAGE_FILE.test(image.file) || !NAME.test(name)) fail(dir, `"${image.file}" must be a kebab-case .webp, .svg, .gif or .png`);
    if (!fs.existsSync(path.join(dir, image.file))) fail(dir, `"${image.file}" is listed in ${INDEX_FILE} but missing`);
    for (const field of ["title", "alt", "width", "height"]) {
      if (!image[field]) fail(dir, `"${image.file}" has no ${field}`);
    }
  }
  for (const f of fs.readdirSync(dir)) {
    if (f !== INDEX_FILE && !listedFiles.has(f)) fail(path.join(dir, f), `not listed in ${INDEX_FILE}`);
  }
}
for (const file of walk(MEDIA_DIR, (p) => IMAGE_FILE.test(p) || p.endsWith(".jpg"))) {
  if (!fs.existsSync(path.join(path.dirname(file), INDEX_FILE))) fail(file, "images live in an image folder with an _index.json");
}

// ---- References in content ----

const assets = fs.readFileSync(path.join(ROOT, "src/assets.ts"), "utf8");
const REMOTE_IMAGE = /https?:\/\/[^\s"'`)]+\.(png|jpe?g|gif|webp|svg)\b/gi;
const sources = ["content", "home", "enterprise"].flatMap((d) => walk(path.join(ROOT, d), (p) => p.endsWith(".mdx")));
for (const file of sources) {
  const text = fs.readFileSync(file, "utf8").replace(/```[\s\S]*?```/g, "");
  for (const [tag] of text.matchAll(/<ImageDisplayer\b[\s\S]*?\/>/g)) {
    const id = /\bid="([^"]+)"/.exec(tag)?.[1];
    if (!id) fail(file, `<ImageDisplayer> takes a library id: ${tag.replace(/\s+/g, " ").slice(0, 80)}`);
    else if (!library.has(id)) fail(file, `unknown media id "${id}"`);
  }
  for (const [url] of text.matchAll(REMOTE_IMAGE)) fail(file, `remote image ${url}; add it to docs/media/ instead`);
  for (const [, key] of text.matchAll(/ASSETS\.(\w+)/g)) {
    if (!new RegExp(`^\\s+${key}:`, "m").test(assets)) fail(file, `ASSETS.${key} doesn't exist in src/assets.ts`);
  }
}

if (errors.length) {
  console.error(`validate-content: ${errors.length} problem(s)\n${errors.map((e) => `  ${e}`).join("\n")}`);
  process.exit(1);
}
console.log(`validate-content: ${posts.length} posts and ${library.size} library images OK`);
