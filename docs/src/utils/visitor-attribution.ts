/**
 * Inbound visitor UTM capture + first/last-touch storage for the deepeval docs
 * site. Mirrors the storage layer of `confident-landing/lib/utm.ts` so that
 * marketing has a consistent attribution model across both surfaces.
 *
 * Schema (stored in localStorage under ATTRIBUTION_STORAGE_KEY):
 *
 *   {
 *     first_touch: { params: { utm_source, utm_medium, ... }, ts: epochMs },
 *     last_touch:  { params: { ... },                           ts: epochMs }
 *   }
 *
 * - `first_touch` is write-once within the TTL window (acquisition channel).
 * - `last_touch`  is overwritten on every capture that contains UTMs.
 * - TTL is checked at READ time; expired touches are treated as absent.
 * - A page load with no UTM params in the URL is a no-op — never clears.
 *
 * All storage access is wrapped in try/catch + `typeof window` guards so this
 * is SSR-safe (Docusaurus runs the click listener client-side, but lifecycle
 * imports may pull this module in during SSR builds).
 */

const UTM_KEYS = [
  'utm_source',
  'utm_medium',
  'utm_campaign',
  'utm_content',
  'utm_term',
] as const;

const ATTRIBUTION_STORAGE_KEY = 'confident_utm_attribution';

const TTL_MS = 180 * 24 * 60 * 60 * 1000;

export type UtmParams = Partial<Record<(typeof UTM_KEYS)[number], string>>;

type Touch = {
  params: UtmParams;
  ts: number;
};

type Attribution = {
  first_touch?: Touch;
  last_touch?: Touch;
};

function safeLocalGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeLocalSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // localStorage unavailable (private browsing, quota, etc.)
  }
}

function isFreshTouch(touch: Touch | undefined): touch is Touch {
  if (!touch || typeof touch.ts !== 'number') return false;
  return Date.now() - touch.ts < TTL_MS;
}

function readAttribution(): Attribution | null {
  if (typeof window === 'undefined') return null;
  const raw = safeLocalGet(ATTRIBUTION_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Attribution;
    if (!parsed || typeof parsed !== 'object') return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeAttribution(attribution: Attribution): void {
  if (typeof window === 'undefined') return;
  try {
    safeLocalSet(ATTRIBUTION_STORAGE_KEY, JSON.stringify(attribution));
  } catch {
    // JSON.stringify shouldn't fail on this shape; guard anyway.
  }
}

/**
 * Read UTM params from the current URL and persist them to localStorage as
 * first_touch (write-once within TTL) + last_touch (always overwrites).
 * No-op when called during SSR or when the URL carries no UTM params.
 */
export function captureVisitorUtms(): void {
  if (typeof window === 'undefined') return;

  const params = new URLSearchParams(window.location.search);
  const utmParams: UtmParams = {};
  for (const key of UTM_KEYS) {
    const value = params.get(key);
    if (value) utmParams[key] = value;
  }
  if (Object.keys(utmParams).length === 0) return;

  const now = Date.now();
  const newTouch: Touch = { params: utmParams, ts: now };
  const existing = readAttribution() ?? {};

  const next: Attribution = {
    first_touch: isFreshTouch(existing.first_touch)
      ? existing.first_touch
      : newTouch,
    last_touch: newTouch,
  };

  writeAttribution(next);
}

export function getFirstTouchParams(): UtmParams | null {
  const attribution = readAttribution();
  if (!attribution || !isFreshTouch(attribution.first_touch)) return null;
  return attribution.first_touch.params;
}

export function getLastTouchParams(): UtmParams | null {
  const attribution = readAttribution();
  if (!attribution || !isFreshTouch(attribution.last_touch)) return null;
  return attribution.last_touch.params;
}

/**
 * The docs pages this tab visited, in order, so links into the app can carry
 * which pages the visitor read as the `site_path` query param. sessionStorage
 * is per tab: a link opened in a new tab starts a new trail. Repeats collapse,
 * each path is cut at 120 characters, and only the entry page plus the last
 * SESSION_PATH_MAX_STEPS - 1 pages survive.
 *
 * The app reads site_path at page load, requires a value that starts with "/",
 * and caps it at 300 characters by cutting the end on a ">" boundary. Keep
 * SITE_PATH_MAX_CHARS at or below that cap, or the app drops the most recent
 * steps. When the joined trail is longer, keep the entry page and the most
 * recent steps with a ">…" segment marking the cut: the entry page says where
 * they landed, the tail says what they read just before the click. The cap
 * counts raw characters, before URL encoding.
 */
const SESSION_PATH_KEY = 'confident_session_path';
const SESSION_PATH_MAX_STEPS = 40;
const SITE_PATH_MAX_CHARS = 300;
const SITE_PATH_GAP = '>…';

function readSessionPath(): string[] {
  try {
    const raw = sessionStorage.getItem(SESSION_PATH_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((p): p is string => typeof p === 'string')
      : [];
  } catch {
    return [];
  }
}

export function recordDocsPageView(pathname: string): void {
  if (typeof window === 'undefined') return;
  const steps = readSessionPath();
  const path = pathname.slice(0, 120);
  if (steps[steps.length - 1] === path) return;
  steps.push(path);
  const kept =
    steps.length > SESSION_PATH_MAX_STEPS
      ? [steps[0], ...steps.slice(-(SESSION_PATH_MAX_STEPS - 1))]
      : steps;
  try {
    sessionStorage.setItem(SESSION_PATH_KEY, JSON.stringify(kept));
  } catch {
    // sessionStorage unavailable
  }
}

export function getDocsSessionPathCompact(): string | undefined {
  if (typeof window === 'undefined') return undefined;
  const paths = readSessionPath();
  if (!paths.length) return undefined;
  const full = paths.join('>');
  if (full.length <= SITE_PATH_MAX_CHARS) return full;
  const head = paths[0];
  let tail = '';
  for (let i = paths.length - 1; i > 0; i--) {
    const next = '>' + paths[i] + tail;
    if (head.length + SITE_PATH_GAP.length + next.length > SITE_PATH_MAX_CHARS)
      break;
    tail = next;
  }
  return (head + SITE_PATH_GAP + tail).slice(0, SITE_PATH_MAX_CHARS);
}
