/**
 * Inbound visitor UTM capture, first/last-touch storage, the per-tab page
 * trail, and the arrival host for the deepeval docs site. The touch storage
 * mirrors `confident-landing/lib/utm.ts` so that marketing has a consistent
 * attribution model across both surfaces. The stored trail shape deliberately
 * differs (see SESSION_PATH_KEY below), but the site_path wire format the
 * trail is sent in is shared with that file (see the marker block below).
 *
 * Schema (stored in localStorage under ATTRIBUTION_STORAGE_KEY):
 *
 *   {
 *     first_touch: { params: { utm_source, utm_medium, ... }, ts: epochMs },
 *     last_touch:  { params: { ... },                           ts: epochMs }
 *   }
 *
 * Schema (stored in sessionStorage under SESSION_PATH_KEY):
 *
 *   ["/docs", "/docs/metrics/answer-relevancy", ...]   pathnames, visit order
 *
 * Schema (stored in sessionStorage under ARRIVAL_KEY):
 *
 *   { host: "google.com" }        the external site this tab arrived from
 *   { host: "" }                  no external referrer: a direct arrival
 *
 * - The arrival is per tab and write-once, evaluated on the first document
 *   load of the tab. It has no TTL; sessionStorage expires it.
 * - The empty host is a recorded answer, not a missing one. A row with no
 *   arrival at all means the capture never ran or storage was blocked, and
 *   the difference is the only way to measure the capture rate.
 *
 * - `first_touch` is write-once within the TTL window (acquisition channel).
 * - `last_touch`  is overwritten on every capture that contains UTMs.
 * - TTL is checked at READ time; expired touches are treated as absent.
 * - A page load with no UTM params in the URL is a no-op — never clears.
 *
 * All storage access is wrapped in try/catch + `typeof window` guards so this
 * is SSR-safe (the Next.js app router runs the click listener client-side,
 * but lifecycle imports may pull this module in during SSR builds).
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
 * The docs pages this tab visited, in order, so a link out of the docs can
 * carry which pages the visitor read as the `site_path` query param.
 * sessionStorage
 * is per tab: a link opened in a new tab starts a new trail, and a tab that
 * leaves the docs and returns keeps its trail. Links into any Confident AI
 * host carry it, behind the arrival markers. Only a page equal to the
 * previous entry is skipped (A > A collapses, A > B > A does not), each path
 * is cut at SESSION_PATH_MAX_PATH_CHARS, and only the entry page plus the
 * last SESSION_PATH_MAX_STEPS - 1 pages survive. The key is docs-specific so
 * the landing site's own trail never collides if both sites ever share an
 * origin: that key is `confident_session_path` and holds objects with clocks
 * rather than plain strings.
 */
const MARKER_PREFIX = '/@';
const SESSION_PATH_KEY = 'confident_docs_session_path';
const SESSION_PATH_MAX_STEPS = 40;
const SESSION_PATH_MAX_PATH_CHARS = 120;
const SITE_PATH_MAX_CHARS = 300;
const SITE_PATH_GAP = '>…';

/**
 * The external site this tab arrived from. Without it, a visitor who reached
 * the docs from a Google result and then signed up records as "deepeval" with
 * nothing before it.
 *
 * The answer is decided once per tab, on the first document load, and the
 * decision is stored even when it is "no external referrer": an empty host is
 * a direct arrival, an absent key means the capture never ran. Binding the
 * decision to the document and not to a successful write is what keeps a
 * later referrer out. A visitor who lands direct, leaves to another site in
 * the same tab and comes back would otherwise have that site recorded as the
 * place the visit started, with the earlier pages already in the trail.
 *
 * Two checks decide "internal". The exact compare against
 * window.location.hostname covers whatever host this build runs on, so
 * preview deployments and localhost never treat a self-referral as an
 * arrival. OWN_HOST_SUFFIXES covers the production domains and their
 * subdomains, which the exact compare alone would miss: a visitor who walks
 * deepeval.com to confident-ai.com and back has not arrived from anywhere new.
 *
 * SUPPRESSED_HOSTS mirrors the backend's OAUTH_REFERRER_PATTERN
 * (confident-cloud apps/backend/src/utils/onboarding-attribution.ts). That
 * rule drops an OAuth referrer from the referrer field; a host recorded here
 * travels in site_path instead, which that rule does not inspect, so the same
 * hosts have to be refused at the source.
 */
const ARRIVAL_KEY = 'confident_docs_arrival';
const OWN_SITE = 'deepeval.com';
const OWN_HOST_SUFFIXES = [OWN_SITE, 'confident-ai.com'];
const SUPPRESSED_HOSTS = /^accounts\.google\./i;
// 100, not the 253 a hostname may legally reach: the arrival sits in front of
// the pages inside a 300-character budget, and a longer value would leave too
// little room to render even the page the visit started on.
const HOST_PATTERN = /^[a-z0-9.-]{1,100}$/;

let arrivalEvaluated = false;

function externalReferrerHost(): string {
  try {
    if (!document.referrer) return '';
    const host = new URL(document.referrer).hostname
      .toLowerCase()
      .replace(/^www\./, '');
    const own = window.location.hostname.toLowerCase().replace(/^www\./, '');
    if (!host || host === own || !HOST_PATTERN.test(host)) return '';
    if (SUPPRESSED_HOSTS.test(host)) return '';
    if (OWN_HOST_SUFFIXES.some((d) => host === d || host.endsWith('.' + d))) {
      return '';
    }
    return host;
  } catch {
    return '';
  }
}

export function captureDocsArrival(): void {
  if (typeof window === 'undefined') return;
  if (arrivalEvaluated) return;
  arrivalEvaluated = true;
  try {
    if (sessionStorage.getItem(ARRIVAL_KEY)) return;
    sessionStorage.setItem(
      ARRIVAL_KEY,
      JSON.stringify({ host: externalReferrerHost() }),
    );
  } catch {
    // sessionStorage unavailable: the arrival stays unknown for this tab
  }
}

/**
 * The stored arrival: a host, '' for a recorded direct arrival, or undefined
 * when nothing was recorded. A stored value that does not validate is dropped
 * rather than kept, so one corrupt write cannot blind the rest of the tab.
 */
function readDocsArrival(): string | undefined {
  try {
    const raw = sessionStorage.getItem(ARRIVAL_KEY);
    if (!raw) return undefined;
    const parsed: unknown = JSON.parse(raw);
    const host =
      parsed && typeof parsed === 'object'
        ? (parsed as { host?: unknown }).host
        : undefined;
    if (typeof host === 'string' && (!host || HOST_PATTERN.test(host))) {
      return host;
    }
    sessionStorage.removeItem(ARRIVAL_KEY);
    return undefined;
  } catch {
    return undefined;
  }
}

function readSessionPath(): string[] {
  try {
    const raw = sessionStorage.getItem(SESSION_PATH_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter(
          (p): p is string => typeof p === 'string' && p.startsWith('/'),
        )
      : [];
  } catch {
    return [];
  }
}

export function recordDocsPageView(pathname: string): void {
  if (typeof window === 'undefined') return;
  const steps = readSessionPath();
  const path = pathname
    .replace(/\/{2,}/g, '/')
    .slice(0, SESSION_PATH_MAX_PATH_CHARS);
  // A real route never starts with "/@", but a 404 does render for one, and a
  // step that looks like a marker would rewrite the trail's own grammar.
  if (path.startsWith(MARKER_PREFIX)) return;
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

/**
 * Trail markers, shared with confident-landing's lib/utm.ts. A step that
 * starts with "/@" is a marker and not a page:
 *
 *   /@from=google.com      the external site the visit started from
 *   /@from=direct          no external referrer, recorded as an answer
 *   /@site=<this site>     the site the steps after it belong to
 *
 * Markers keep the whole value starting with "/", which is what the app
 * requires, and let one trail carry steps from both sites once the landing
 * site seeds its own trail from this value.
 */
const DIRECT_ARRIVAL = 'direct';

function trailMarkers(): string[] {
  const markers: string[] = [];
  const arrival = readDocsArrival();
  if (arrival !== undefined) {
    markers.push(`${MARKER_PREFIX}from=${arrival || DIRECT_ARRIVAL}`);
  }
  markers.push(`${MARKER_PREFIX}site=${OWN_SITE}`);
  return markers;
}

/**
 * The trail as one query value: the markers, then the paths, joined by ">".
 * App-side contract
 * (confident-cloud apps/frontend/src/utils/utm.ts,
 * MAX_ATTRIBUTION_VALUE_LENGTH): the app reads site_path at page load, keeps
 * it only when it starts with "/" (and not "//"), and cuts anything past 300
 * characters back to the last ">" boundary. Keep SITE_PATH_MAX_CHARS at or
 * below that value, or the app drops the most recent steps. When the joined
 * trail is longer, keep the entry page and the most recent steps with a
 * SITE_PATH_GAP segment marking the cut: the entry page says where they
 * landed, the tail says what they read just before the click. The cap counts
 * raw characters, before URL encoding.
 */
export function getDocsSessionPathCompact(): string | undefined {
  if (typeof window === 'undefined') return undefined;
  // The markers are built before the trail is read, so an arrival still
  // travels when the trail is empty or unreadable. The two stores fail
  // independently, and the arrival is the smaller loss to accept.
  const markers = trailMarkers().join('>');
  const paths = readSessionPath();
  if (!paths.length) return markers || undefined;
  // Markers go in front and are never cut, because the truncation below only
  // drops from the tail. There are one or two of them, and HOST_PATTERN caps
  // the arrival host at 100 characters, so the prefix stays well inside the
  // budget. They carry the only answer to "where was this visitor before
  // deepeval.com" when the visit arrived with no utm params.
  const prefix = markers + '>';
  const budget = SITE_PATH_MAX_CHARS - prefix.length;
  const full = paths.join('>');
  if (full.length <= budget) return prefix + full;
  const head = paths[0];
  let tail = '';
  for (let i = paths.length - 1; i > 0; i--) {
    const next = '>' + paths[i] + tail;
    if (head.length + SITE_PATH_GAP.length + next.length > budget) break;
    tail = next;
  }
  return (prefix + head + SITE_PATH_GAP + tail).slice(0, SITE_PATH_MAX_CHARS);
}
