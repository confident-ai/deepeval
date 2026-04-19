/**
 * UTM tagging primitives for outbound links from the deepeval docs site to
 * Confident AI properties.
 *
 * The site is on a PURE-RUNTIME tagging model:
 *   - For anchor clicks, the click-time interceptor in
 *     `src/clientModules/confident-ref-page.ts` walks every <a href> at click
 *     time, and (if the host is in CONFIDENT_HOSTNAMES) stamps the full UTM
 *     payload onto the href before the browser navigates. utm_content is
 *     resolved from the anchor's `data-utm-content` attribute or a
 *     `utm--<value>` className convention; falls back to "inline_link".
 *
 *   - For IMPERATIVE navigations (window.open / location.href = ...) there is
 *     no anchor for the click listener to grab. Those callsites use the
 *     `appendDeepEvalAttribution` helper exported below to do the same UTM
 *     assembly synchronously at call time.
 *
 * Schema (matches click-listener output):
 *   utm_source  = "deepeval"               (constant, deepeval-owned)
 *   utm_medium  = "docs" | "github" | "cli" | "python_sdk"
 *                                          (constant; default "docs")
 *   utm_content = location on the source surface (e.g. "navbar",
 *                 "video_overlay")        (deepeval-owned, per CTA)
 *   utm_campaign = inbound visitor campaign carried via last_touch
 *   utm_term     = inbound visitor term carried via last_touch
 *   ref_page    = window.location.pathname at click time (always)
 *
 * `utm_campaign` and `utm_term` come from the visitor's stored last_touch
 * (captured from the URL when they first landed). This lets a Google ad
 * campaign survive the deepeval-docs hop into app.confident-ai.com.
 *
 * Programmatic hosts (api.*, eu.api.*, au.api.*, deepeval.*, eu.deepeval.*,
 * au.deepeval.*, otel.*, eu.otel.*, au.otel.*) are intentionally excluded
 * — they're API/OTel endpoints, not browser-clickable.
 */

import { getLastTouchParams } from './visitor-attribution';

/**
 * The three browser-clickable Confident AI hosts. Single source of truth for
 * any code building a link.
 */
export const CONFIDENT_HOSTS_BY_NAME = {
  /** Marketing site + cloud docs. The default for nearly every link. */
  WWW: 'https://www.confident-ai.com',
  /** App dashboard / sign-up. */
  APP: 'https://app.confident-ai.com',
  /** Marketing root (no `www.` prefix). */
  ROOT: 'https://confident-ai.com',
} as const;

export type ConfidentHost = keyof typeof CONFIDENT_HOSTS_BY_NAME;

/** Hostname-only set used by both runtime guards (URL#hostname comparison). */
export const CONFIDENT_HOSTNAMES: ReadonlySet<string> = new Set(
  Object.values(CONFIDENT_HOSTS_BY_NAME).map((u) => new URL(u).hostname),
);

const SOURCE = 'deepeval';
const DEFAULT_MEDIUM = 'docs';

export type UtmMedium = 'docs' | 'github' | 'cli' | 'python_sdk';

export interface AppendOpts {
  /** Location on the source surface (e.g. "video_overlay", "sidebar_promo"). */
  content: string;
  /** Surface type. Defaults to "docs". */
  medium?: UtmMedium;
}

/**
 * Stamp the full deepeval UTM payload onto a Confident AI URL at runtime.
 * Use from imperative-navigation callsites (window.open / location.href = ...)
 * where there's no anchor element for the click-time interceptor to see.
 *
 * No-op (returns input unchanged) if:
 *   - URL is not a string or is empty
 *   - URL cannot be parsed as a URL
 *   - URL is not a Confident AI host (CONFIDENT_HOSTNAMES)
 *   - The corresponding param is already set on the URL (caller wins)
 *
 * Browser-only — pulls last_touch from localStorage and current pathname from
 * window.location. SSR-safe (returns input unchanged when window is undefined).
 */
export function appendDeepEvalAttribution(
  url: string,
  opts: AppendOpts,
): string {
  if (typeof url !== 'string' || !url) return url;
  if (typeof window === 'undefined') return url;

  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return url;
  }
  if (!CONFIDENT_HOSTNAMES.has(u.hostname)) return url;

  const { content, medium = DEFAULT_MEDIUM } = opts;

  if (!u.searchParams.has('utm_source')) u.searchParams.set('utm_source', SOURCE);
  if (!u.searchParams.has('utm_medium')) u.searchParams.set('utm_medium', medium);
  if (content && !u.searchParams.has('utm_content')) {
    u.searchParams.set('utm_content', content);
  }

  const last = getLastTouchParams();
  if (last) {
    if (last.utm_campaign && !u.searchParams.has('utm_campaign')) {
      u.searchParams.set('utm_campaign', last.utm_campaign);
    }
    if (last.utm_term && !u.searchParams.has('utm_term')) {
      u.searchParams.set('utm_term', last.utm_term);
    }
  }

  if (!u.searchParams.has('ref_page')) {
    u.searchParams.set('ref_page', window.location.pathname);
  }

  return u.toString();
}
