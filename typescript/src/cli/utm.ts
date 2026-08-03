// UTM tagging for browser-clickable Confident AI links, mirroring `with_utm` in
// deepeval/cli/utils.py. No `ref_page`: a CLI invocation is not a page.
// Programmatic hosts (api.*, deepeval.*, otel.*) are never tagged.

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

/** Open a URL in the user's browser, ignoring failures (headless, SSH, CI). */
export async function openBrowser(url: string): Promise<void> {
  try {
    const open = (await import("open")).default;
    await open(url);
  } catch {}
}
