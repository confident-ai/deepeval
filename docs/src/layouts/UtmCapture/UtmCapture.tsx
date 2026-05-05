"use client";

/**
 * Global UTM tagging for the deepeval docs site. Mirrors the runtime-only
 * architecture used by confident-landing's <UtmCapture> (see
 * confident-landing/components/UtmCapture/UtmCapture.tsx).
 *
 * ─── Two responsibilities ───────────────────────────────────────────────────
 *
 * 1. INBOUND: on mount (and on SPA route changes), call captureVisitorUtms to
 *    read the current URL's `utm_*` params and persist them in localStorage as
 *    first_touch (write-once within 180-day TTL) + last_touch (overwrites).
 *    This lets the original Google / LinkedIn / etc. campaign survive the
 *    deepeval-docs hop into app.confident-ai.com at signup.
 *
 * 2. OUTBOUND: on every click that targets a Confident AI host, stamp the full
 *    UTM payload onto the anchor's href before the browser navigates.
 *
 *      utm_source   = "deepeval"                  (constant)
 *      utm_medium   = anchor data-utm-medium ?? "docs"
 *      utm_content  = anchor data-utm-content,
 *                     or className matching `utm--<value>`,
 *                     or "inline_link" (fallback for MDX body links)
 *      utm_campaign = last_touch.utm_campaign     (visitor-derived)
 *      utm_term     = last_touch.utm_term         (visitor-derived)
 *      ref_page     = window.location.pathname
 *
 *    Caller-set params on the existing href are preserved (we never clobber).
 *
 * ─── Why click-time, not React event handlers ───────────────────────────────
 * Document-level capture-phase listeners on `mousedown`, `auxclick`, and
 * `keydown` (Enter). For each event we walk up from `e.target` to the nearest
 * enclosing `<a href>`, parse the href, and stamp params before navigation.
 *
 *   - mousedown: fires before navigation on a primary-button click.
 *   - auxclick:  middle-click and cmd/ctrl-click "open in new tab".
 *   - keydown:   Enter on a focused link.
 *
 * Capture phase ensures we run before any Next.js framework navigation
 * handler. Event delegation means the patch covers links that don't exist yet
 * — SPA navigations swap content without rebinding.
 *
 * ─── What it intentionally does NOT touch ───────────────────────────────────
 * Imperative navigations (window.open / window.location.href = ...) bypass
 * any anchor element, so they never reach this listener. Components that do
 * imperative navigation must call appendDeepEvalAttribution() from
 * src/utils/utm.ts directly (VideoDisplayer, etc.).
 *
 * Mounted once in app/layout.tsx.
 */

import { useEffect } from "react";
import {
  CONFIDENT_HOSTNAMES,
  type UtmMedium,
} from "@/src/utils/utm";
import {
  captureVisitorUtms,
  getLastTouchParams,
} from "@/src/utils/visitor-attribution";

const SOURCE = "deepeval";
const DEFAULT_MEDIUM: UtmMedium = "docs";
const CLASS_PREFIX = "utm--";
const FALLBACK_CONTENT = "inline_link";

function resolveUtmContent(anchor: HTMLAnchorElement): string {
  const explicit =
    anchor.getAttribute("data-utm-content") ??
    anchor.closest("[data-utm-content]")?.getAttribute("data-utm-content");
  if (explicit) return explicit;

  for (const cls of Array.from(anchor.classList)) {
    if (cls.startsWith(CLASS_PREFIX)) return cls.slice(CLASS_PREFIX.length);
  }

  let parent: Element | null = anchor.parentElement;
  while (parent) {
    for (const cls of Array.from(parent.classList)) {
      if (cls.startsWith(CLASS_PREFIX)) return cls.slice(CLASS_PREFIX.length);
    }
    parent = parent.parentElement;
  }

  return FALLBACK_CONTENT;
}

function stampAnchor(anchor: HTMLAnchorElement | null): void {
  if (!anchor) return;
  const href = anchor.getAttribute("href");
  if (!href) return;

  let u: URL;
  try {
    u = new URL(href, window.location.href);
  } catch {
    return;
  }
  if (!CONFIDENT_HOSTNAMES.has(u.hostname)) return;

  const content = resolveUtmContent(anchor);
  const medium =
    (anchor.getAttribute("data-utm-medium") as UtmMedium | null) ??
    DEFAULT_MEDIUM;

  if (!u.searchParams.has("utm_source"))
    u.searchParams.set("utm_source", SOURCE);
  if (!u.searchParams.has("utm_medium"))
    u.searchParams.set("utm_medium", medium);
  if (content && !u.searchParams.has("utm_content")) {
    u.searchParams.set("utm_content", content);
  }

  const last = getLastTouchParams();
  if (last) {
    if (last.utm_campaign && !u.searchParams.has("utm_campaign")) {
      u.searchParams.set("utm_campaign", last.utm_campaign);
    }
    if (last.utm_term && !u.searchParams.has("utm_term")) {
      u.searchParams.set("utm_term", last.utm_term);
    }
  }

  if (!u.searchParams.has("ref_page")) {
    u.searchParams.set("ref_page", window.location.pathname);
  }

  anchor.setAttribute("href", u.toString());
}

function handleEvent(e: Event): void {
  const target = e.target as Element | null;
  const anchor =
    target && typeof target.closest === "function"
      ? (target.closest("a[href]") as HTMLAnchorElement | null)
      : null;
  stampAnchor(anchor);
}

const UtmCapture = () => {
  useEffect(() => {
    captureVisitorUtms();

    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function patchedPushState(
      this: History,
      ...args: Parameters<History["pushState"]>
    ) {
      const result = originalPushState.apply(this, args);
      try {
        captureVisitorUtms();
      } catch {
        // never let attribution errors break navigation
      }
      return result;
    };

    history.replaceState = function patchedReplaceState(
      this: History,
      ...args: Parameters<History["replaceState"]>
    ) {
      const result = originalReplaceState.apply(this, args);
      try {
        captureVisitorUtms();
      } catch {
        // never let attribution errors break navigation
      }
      return result;
    };

    const handlePopState = () => {
      try {
        captureVisitorUtms();
      } catch {
        // swallow — see above
      }
    };

    const handleKeydown = (e: KeyboardEvent) => {
      if (e.key === "Enter") handleEvent(e);
    };

    window.addEventListener("popstate", handlePopState);
    document.addEventListener("mousedown", handleEvent, true);
    document.addEventListener("auxclick", handleEvent, true);
    document.addEventListener("keydown", handleKeydown, true);

    return () => {
      history.pushState = originalPushState;
      history.replaceState = originalReplaceState;
      window.removeEventListener("popstate", handlePopState);
      document.removeEventListener("mousedown", handleEvent, true);
      document.removeEventListener("auxclick", handleEvent, true);
      document.removeEventListener("keydown", handleKeydown, true);
    };
  }, []);

  return null;
};

export default UtmCapture;
