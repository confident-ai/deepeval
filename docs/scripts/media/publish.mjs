#!/usr/bin/env node
/**
 * Copies the media library and blog covers into `public/media/` (gitignored),
 * where they're served from: `/media/<id>.<ext>` and `/media/covers/<slug>.<ext>`.
 */
import fs from "node:fs";
import path from "node:path";
import { MEDIA_DIR, loadLibrary } from "../../lib/media/library.mjs";
import { BLOG_DIR, COVER_FILE, MONTH_FOLDER } from "../../lib/post-folders.mjs";

const OUT = path.join(process.cwd(), "public/media");

fs.rmSync(OUT, { recursive: true, force: true });

let images = 0;
for (const image of loadLibrary().values()) {
  const to = path.join(OUT, image.folder, image.file);
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(path.join(MEDIA_DIR, image.folder, image.file), to);
  images++;
}

let covers = 0;
fs.mkdirSync(path.join(OUT, "covers"), { recursive: true });
for (const month of fs.readdirSync(BLOG_DIR).filter((m) => MONTH_FOLDER.test(m))) {
  for (const slug of fs.readdirSync(path.join(BLOG_DIR, month))) {
    const dir = path.join(BLOG_DIR, month, slug);
    const cover = fs.readdirSync(dir).find((f) => COVER_FILE.test(f));
    if (!cover) continue;
    fs.copyFileSync(path.join(dir, cover), path.join(OUT, "covers", `${slug}${path.extname(cover)}`));
    covers++;
  }
}

console.log(`media: published ${images} images and ${covers} covers to public/media/`);
