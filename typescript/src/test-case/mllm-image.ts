import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

// Mirrors deepeval's MLLMImage + the slug/registry machinery. Images embed in
// normal string fields as `[DEEPEVAL:IMAGE|PDF:{id}]` slugs (via toString) and
// are looked back up by id from a global registry.

export const SLUG_PATTERN = /\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]/;
export const SLUG_PATTERN_G = /\[DEEPEVAL:(?:IMAGE|PDF):(.*?)\]/g;

/** Global id → image registry (mirrors Python's `_MLLM_IMAGE_REGISTRY`). */
export const MLLM_IMAGE_REGISTRY = new Map<string, MLLMImage>();

const MIME_BY_EXT: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".svg": "image/svg+xml",
  ".tiff": "image/tiff",
  ".tif": "image/tiff",
  ".pdf": "application/pdf",
};

function guessMimeType(filename: string): string | undefined {
  return MIME_BY_EXT[path.extname(filename).toLowerCase()];
}

export interface MLLMImageParams {
  url?: string;
  dataBase64?: string;
  mimeType?: string;
  local?: boolean;
  filename?: string;
  id?: string;
}

export class MLLMImage {
  dataBase64?: string;
  mimeType?: string;
  url?: string;
  local?: boolean;
  filename?: string;
  id: string;

  constructor(params: MLLMImageParams = {}) {
    this.url = params.url;
    this.dataBase64 = params.dataBase64;
    this.mimeType = params.mimeType;
    this.local = params.local;
    this.filename = params.filename;
    this.id = params.id ?? randomUUID().replace(/-/g, "");

    if (!this.url && !this.dataBase64) {
      throw new Error(
        "You must provide either a 'url' or both 'dataBase64' and 'mimeType' to create an MLLMImage.",
      );
    }

    if (this.dataBase64 != null) {
      if (this.mimeType == null) {
        throw new Error(
          "mimeType must be provided when initializing from Base64 data.",
        );
      }
    } else {
      const url = this.url as string;
      const isLocal = MLLMImage.isLocalPath(url);
      if (this.local != null) {
        if (this.local !== isLocal) throw new Error("Local path mismatch");
      } else {
        this.local = isLocal;
      }

      if (this.local) {
        const p = MLLMImage.processUrl(url);
        this.filename = path.basename(p);
        this.mimeType = guessMimeType(p) ?? "image/jpeg";
        if (!fs.existsSync(p)) {
          throw new Error(`Image file not found: ${p}`);
        }
        this.loadBase64(p);
      } else {
        if (!(url.startsWith("http://") || url.startsWith("https://"))) {
          throw new Error(
            `Invalid remote URL format: ${url}. URL must start with http:// or https://`,
          );
        }
        this.filename = path.basename(new URL(url).pathname);
        this.mimeType = guessMimeType(this.filename);
        this.dataBase64 = undefined;
      }
    }

    MLLM_IMAGE_REGISTRY.set(this.id, this);
  }

  private loadBase64(p: string): void {
    this.dataBase64 = fs.readFileSync(p).toString("base64");
  }

  ensureImagesLoaded(): this {
    if (this.local && this.dataBase64 == null && this.url) {
      this.loadBase64(MLLMImage.processUrl(this.url));
    }
    return this;
  }

  private placeholder(): string {
    return this.mimeType === "application/pdf"
      ? `[DEEPEVAL:PDF:${this.id}]`
      : `[DEEPEVAL:IMAGE:${this.id}]`;
  }

  /** The slug — so `\`...${image}\`` (template literals) embeds the image, like Python f-strings. */
  toString(): string {
    return this.placeholder();
  }

  asDataUri(): string | undefined {
    if (!this.dataBase64 || !this.mimeType) return undefined;
    return `data:${this.mimeType};base64,${this.dataBase64}`;
  }

  static processUrl(url: string): string {
    if (fs.existsSync(url)) return url;
    if (url.startsWith("file:")) {
      try {
        return fileURLToPath(url);
      } catch {
        return url;
      }
    }
    return url;
  }

  static isLocalPath(url: string): boolean {
    if (fs.existsSync(url)) return true;
    if (url.startsWith("file:")) {
      try {
        return fs.existsSync(fileURLToPath(url));
      } catch {
        return false;
      }
    }
    return false;
  }

  /** Split a string into its text + image parts (mirrors `parse_multimodal_string`). */
  static parseMultimodalString(s: string): (string | MLLMImage)[] {
    const result: (string | MLLMImage)[] = [];
    let lastEnd = 0;
    for (const m of s.matchAll(SLUG_PATTERN_G)) {
      const start = m.index ?? 0;
      const end = start + m[0].length;
      if (start > lastEnd) result.push(s.slice(lastEnd, start));
      const imgId = m[1];
      const img =
        MLLM_IMAGE_REGISTRY.get(imgId) ??
        new MLLMImage({ url: imgId, id: imgId });
      result.push(img);
      lastEnd = end;
    }
    if (lastEnd < s.length) result.push(s.slice(lastEnd));
    return result;
  }
}

/** True if the string contains an image/pdf slug. */
export function checkIfMultimodal(input: string): boolean {
  return SLUG_PATTERN.test(input);
}

/** Convert a string (or list of strings) into a flat multimodal array. */
export function convertToMultiModalArray(
  input: string | string[],
): (string | MLLMImage)[] {
  if (typeof input === "string") return MLLMImage.parseMultimodalString(input);
  return input.flatMap((c) => MLLMImage.parseMultimodalString(c));
}

export function extractImageIdsFromString(
  s: string | undefined | null,
  targetSet: Set<string>,
): void {
  if (!s) return;
  for (const m of s.matchAll(SLUG_PATTERN_G)) {
    targetSet.add(m[1]);
  }
}

export function extractImageIdsFromList(
  lst: (string | undefined | null)[] | undefined | null,
  targetSet: Set<string>,
): void {
  if (!lst) return;
  for (const item of lst) {
    extractImageIdsFromString(item, targetSet);
  }
}