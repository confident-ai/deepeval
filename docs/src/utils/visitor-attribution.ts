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
