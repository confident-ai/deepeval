// UTM tagging for browser-clickable Confident AI links, mirroring `with_utm` in
// deepeval/cli/utils.py. No `ref_page`: a CLI invocation is not a page.
// Programmatic hosts (api.*, deepeval.*, otel.*) are never tagged.

import { parseBool } from "@/config/utils";

export const PROD = "https://app.confident-ai.com";
export const WWW = "https://www.confident-ai.com";

const TAGGED_HOSTS = new Set([
  "confident-ai.com",
  "www.confident-ai.com",
  "app.confident-ai.com",
]);

const UTM_SOURCE = "deepeval";

export function withUtm(
  url: string,
  { medium = "cli", content }: { medium?: string; content: string },
): string {
  if (!url) return url;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  if (!TAGGED_HOSTS.has(parsed.hostname)) return url;
  // Don't clobber tagging that came from upstream.
  if (parsed.searchParams.has("utm_source")) return url;

  parsed.searchParams.set("utm_source", UTM_SOURCE);
  parsed.searchParams.set("utm_medium", medium);
  parsed.searchParams.set("utm_content", content);
  return parsed.toString();
}

/**
 * Ignores failures (headless, SSH, CI). Suppressed by
 * `CONFIDENT_OPEN_BROWSER=0`, which defaults to on.
 */
export async function openBrowser(url: string): Promise<void> {
  if (!(parseBool(process.env.CONFIDENT_OPEN_BROWSER) ?? true)) return;
  try {
    const open = (await import("open")).default;
    await open(url);
  } catch {}
}
