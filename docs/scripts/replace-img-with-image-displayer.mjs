#!/usr/bin/env node
/**
 * Replace raw <img ... /> tags and ![alt](url) markdown images in MDX
 * content with <ImageDisplayer src=... alt=... />.
 *
 *   yarn node scripts/replace-img-with-image-displayer.mjs
 *
 * - Preserves src (string literals or {jsx} expressions) and alt.
 * - Drops other attributes (style, id, width, etc.) — the component is
 *   intentionally a "simple image tag" for now.
 * - Skips content inside fenced code blocks (``` ... ```).
 */
import { readdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { join } from "node:path";

const IMG_TAG = /<img\b([^>]*?)\/?\s*>/gs;
const MD_IMG = /!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
const FENCE = /```[\s\S]*?```/g;

function extractAttr(attrs, name) {
  const jsx = new RegExp(String.raw`\b${name}\s*=\s*\{([^}]*)\}`).exec(attrs);
  if (jsx) return { kind: "jsx", value: jsx[1].trim() };
  const dq = new RegExp(String.raw`\b${name}\s*=\s*"([^"]*)"`).exec(attrs);
  if (dq) return { kind: "string", value: dq[1] };
  const sq = new RegExp(String.raw`\b${name}\s*=\s*'([^']*)'`).exec(attrs);
  if (sq) return { kind: "string", value: sq[1] };
  return null;
}

function formatAttr(name, attr) {
  if (!attr) return "";
  if (attr.kind === "jsx") return `${name}={${attr.value}}`;
  return `${name}="${attr.value.replace(/"/g, '&quot;')}"`;
}

function replaceImgTag(_match, attrs) {
  const src = extractAttr(attrs, "src");
  if (!src) return _match; // leave as-is if we can't find a src
  const alt = extractAttr(attrs, "alt");
  const parts = [formatAttr("src", src)];
  const altStr = formatAttr("alt", alt);
  if (altStr) parts.push(altStr);
  return `<ImageDisplayer ${parts.join(" ")} />`;
}

function replaceMarkdownImg(_match, alt, url) {
  const parts = [`src="${url}"`];
  if (alt) parts.push(`alt="${alt.replace(/"/g, '&quot;')}"`);
  return `<ImageDisplayer ${parts.join(" ")} />`;
}

function transform(src) {
  // Split out fenced code blocks so we don't rewrite examples.
  const chunks = [];
  let last = 0;
  for (const m of src.matchAll(FENCE)) {
    chunks.push({ text: src.slice(last, m.index), code: false });
    chunks.push({ text: m[0], code: true });
    last = m.index + m[0].length;
  }
  chunks.push({ text: src.slice(last), code: false });

  return chunks
    .map((c) =>
      c.code
        ? c.text
        : c.text.replace(IMG_TAG, replaceImgTag).replace(MD_IMG, replaceMarkdownImg),
    )
    .join("");
}

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const s = statSync(full);
    if (s.isDirectory()) walk(full);
    else if (full.endsWith(".mdx")) processFile(full);
  }
}

let changed = 0;
function processFile(path) {
  const src = readFileSync(path, "utf8");
  const out = transform(src);
  if (out !== src) {
    writeFileSync(path, out);
    changed += 1;
    console.log("·", path);
  }
}

walk("content");
console.log(`\n${changed} file(s) updated.`);
