import fs from "node:fs";
import path from "node:path";

/**
 * The media library: every content image lives in git under `media/`, listed
 * in its folder's `_index.json`. An image's id is its path without the
 * extension (`diagrams/in-house/evaluation/llm-test-case`), and it's served
 * from `/media/<path>` once `scripts/media/publish.mjs` has copied it.
 */
export const MEDIA_DIR = path.join(process.cwd(), "media");
export const INDEX_FILE = "_index.json";
export const IMAGE_FILE = /\.(webp|svg|gif|png)$/;
export const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** The folders that hold images. Add a folder here before putting images in it. */
export const IMAGE_FOLDERS = [
  "diagrams/in-house/agents",
  "diagrams/in-house/datasets",
  "diagrams/in-house/evaluation",
  "diagrams/in-house/metrics",
  "diagrams/in-house/rag",
  "diagrams/in-house/red-teaming",
  "diagrams/in-house/tracing",
  "diagrams/external/reference",
  "platform/annotation",
  "platform/datasets",
  "platform/evaluation",
  "platform/monitoring",
  "platform/prompts",
  "platform/settings",
  "platform/tracing",
  "screenshots/deepeval",
  "screenshots/example-apps",
];

/** Every `_index.json` under `media/`, as folder paths relative to it. */
export function indexFolders() {
  const folders = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) walk(path.join(dir, entry.name));
      else if (entry.name === INDEX_FILE) folders.push(path.relative(MEDIA_DIR, dir));
    }
  };
  if (fs.existsSync(MEDIA_DIR)) walk(MEDIA_DIR);
  return folders.sort();
}

let cache;

/** Map of id → { id, folder, file, title, alt, width, height }. Re-read in dev so new images show up. */
export function loadLibrary() {
  if (cache && process.env.NODE_ENV === "production") return cache;
  const images = new Map();
  for (const folder of indexFolders()) {
    const index = JSON.parse(fs.readFileSync(path.join(MEDIA_DIR, folder, INDEX_FILE), "utf8"));
    for (const image of index.images) {
      const id = `${folder}/${path.basename(image.file, path.extname(image.file))}`;
      images.set(id, { id, folder, ...image });
    }
  }
  cache = images;
  return images;
}

/** What an `<img>` needs for a library id; throws on an unknown id. */
export function resolveMedia(id) {
  const image = loadLibrary().get(id);
  if (!image) throw new Error(`Unknown media id "${id}". Ids are paths under docs/media/ without the extension.`);
  return {
    src: `/media/${image.folder}/${image.file}`,
    alt: image.alt,
    width: image.width,
    height: image.height,
  };
}
