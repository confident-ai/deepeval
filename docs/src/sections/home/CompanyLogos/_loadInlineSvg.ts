import fs from "node:fs";
import path from "node:path";

export type LoadedSvg = {
  inner: string;
  viewBox: string | undefined;
  rootFill: string | undefined;
  rootStroke: string | undefined;
};

/**
 * Reads a brand SVG from `/public/icons/companies` at module-init time and
 * returns its inner markup + viewBox so it can be inlined into the SSR HTML
 * response. This is what the Mercedes-Benz logo already does manually — we
 * just generalize it so every brand mark we don't hand-author can ride the
 * same path and avoid the per-image HTTP round trip that `next/image` adds.
 *
 * Two pieces of cleanup are required before the markup is safe to drop into
 * an inline `<svg>` via `dangerouslySetInnerHTML`:
 *
 *   1. XML prologue / DOCTYPE / CDATA wrappers / HTML comments are stripped
 *      because the HTML5 parser used by `innerHTML` doesn't understand them
 *      and would either expose them as text or silently drop the contained
 *      `<style>` rules (Siemens hits the CDATA case).
 *   2. Any CSS class names referenced inside the SVG are prefixed with the
 *      slug so that multiple inline SVGs sharing generic selectors like
 *      `.cls-1` (AXA) or `.fil0` (Siemens) can't fight each other once
 *      they're both living in the same document.
 */
export function loadInlineSvg(slug: string): LoadedSvg {
  const filePath = path.join(
    process.cwd(),
    "public",
    "icons",
    "companies",
    `${slug}.svg`,
  );
  let raw = fs.readFileSync(filePath, "utf8");

  raw = raw
    .replace(/<\?xml[^?]*\?>/g, "")
    .replace(/<!DOCTYPE[^>]*>/g, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<!\[CDATA\[/g, "")
    .replace(/\]\]>/g, "");

  const seen = new Set<string>();
  for (const match of raw.matchAll(/class="([^"]+)"/g)) {
    for (const name of match[1].split(/\s+/)) {
      if (name) seen.add(name);
    }
  }
  if (seen.size > 0) {
    const prefix = `__${slug.replace(/[^a-zA-Z0-9]/g, "_")}_`;
    for (const name of seen) {
      const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const scoped = `${prefix}${name}`;
      raw = raw
        .replace(
          new RegExp(`class="${escaped}"`, "g"),
          `class="${scoped}"`,
        )
        .replace(new RegExp(`\\.${escaped}\\b`, "g"), `.${scoped}`);
    }
  }

  const svgMatch = raw.match(/<svg([^>]*)>([\s\S]*)<\/svg>/);
  const rootAttrs = svgMatch?.[1] ?? "";
  const inner = svgMatch?.[2] ?? "";

  const viewBoxMatch = rootAttrs.match(/viewBox="([^"]+)"/);
  const widthMatch = rootAttrs.match(/\swidth="([\d.]+)/);
  const heightMatch = rootAttrs.match(/\sheight="([\d.]+)/);
  const viewBox =
    viewBoxMatch?.[1] ??
    (widthMatch && heightMatch
      ? `0 0 ${widthMatch[1]} ${heightMatch[1]}`
      : undefined);

  // Some brand SVGs (e.g. Adobe) declare their fill on the root <svg> and
  // leave the child <path>s with no fill of their own. Capture those so the
  // wrapper component can re-apply them — otherwise the logo paints in the
  // default black instead of its brand color.
  const rootFill = rootAttrs.match(/\sfill="([^"]+)"/)?.[1];
  const rootStroke = rootAttrs.match(/\sstroke="([^"]+)"/)?.[1];

  return { inner, viewBox, rootFill, rootStroke };
}
