// Shared rules for docs/lib/community/events.json. Used by the prebuild
// check (scripts/validate-content.mjs) and by the approval workflow
// (scripts/community/add-event.mjs), so a bad entry fails the same way
// in both places.

export const EVENTS_FILE = "lib/community/events.json";

// New submissions are Luma only: we see the guest list through the
// co-host invite, and Luma is the only platform we check for it.
export const LUMA_URL = /^https:\/\/(lu\.ma|luma\.com)\/[A-Za-z0-9-]{3,64}$/;
export const EVENT_ID = /^\d{4}-\d{2}-\d{2}-[a-z0-9]+(-[a-z0-9]+)*$/;
export const GITHUB_LOGIN = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
// Covers live under public/community/ (the approval script writes <id>.jpg;
// a maintainer may add a png or webp by hand) and are never loaded from Luma
// at view time.
export const IMAGE_PATH = /^\/community\/[a-z0-9-]+\.(jpg|png|webp)$/;
// ISO 8601 with an explicit offset, e.g. 2026-11-20T17:00:00-08:00.
export const START = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$/;

const TEXT_LIMITS = { title: 100, city: 60, country: 60, talk: 140, host: 80 };

/** True for a real YYYY-MM-DD date. Date.parse accepts 2026-02-31, so round-trip it. */
export function isCalendarDate(date) {
  if (typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return false;
  const [y, m, d] = date.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toISOString().slice(0, 10) === date;
}

export function isTimeZone(tz) {
  if (typeof tz !== "string" || !tz) return false;
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: tz });
    return true;
  } catch {
    return false;
  }
}

/** Returns a list of problems for one event. Empty means valid. */
export function validateEvent(e) {
  const out = [];
  if (!e || typeof e !== "object" || Array.isArray(e)) return ["an event must be an object"];
  if (!EVENT_ID.test(e.id ?? "")) out.push("id must look like 2026-11-20-san-francisco");
  for (const [key, max] of Object.entries(TEXT_LIMITS)) {
    const v = e[key];
    if (typeof v !== "string" || !v.trim()) out.push(`${key} is required`);
    else if (v.length > max) out.push(`${key} must be ${max} characters or less`);
    else if (/[<>]/.test(v)) out.push(`${key} must not contain < or >`);
  }
  if (!START.test(e.start ?? "") || Number.isNaN(Date.parse(e.start)) || !isCalendarDate(e.start.slice(0, 10))) {
    out.push("start must be ISO 8601 with an offset, e.g. 2026-11-20T17:00:00-08:00");
  }
  if (!isTimeZone(e.timezone)) out.push("timezone must be an IANA name, e.g. Europe/London");
  if (!LUMA_URL.test(e.url ?? "")) out.push("url must be a Luma event link, e.g. https://luma.com/abc123");
  if (typeof e.official !== "boolean") out.push("official must be true or false");
  if (e.hostGithub !== undefined && !GITHUB_LOGIN.test(e.hostGithub)) out.push("hostGithub must be a GitHub username");
  if (e.image !== undefined && !IMAGE_PATH.test(e.image)) out.push("image must be /community/<id>.jpg|png|webp");
  // Optional; an event with both places a mascot on the community globe.
  if (e.lat !== undefined && !(typeof e.lat === "number" && e.lat >= -90 && e.lat <= 90)) out.push("lat must be a number from -90 to 90");
  if (e.lng !== undefined && !(typeof e.lng === "number" && e.lng >= -180 && e.lng <= 180)) out.push("lng must be a number from -180 to 180");
  if ((e.lat === undefined) !== (e.lng === undefined)) out.push("lat and lng must be set together");
  if (e.issue !== undefined && !(Number.isInteger(e.issue) && e.issue > 0)) out.push("issue must be a positive integer");
  const allowed = new Set(["id", "title", "start", "timezone", "city", "country", "url", "talk", "host", "hostGithub", "official", "issue", "image", "lat", "lng"]);
  for (const key of Object.keys(e)) if (!allowed.has(key)) out.push(`unknown field "${key}"`);
  return out;
}

/** Returns a list of problems for the whole file. Empty means valid. */
export function validateEvents(list) {
  if (!Array.isArray(list)) return ["the file must hold a JSON array"];
  const out = [];
  const ids = new Set();
  const urls = new Set();
  list.forEach((e, i) => {
    const label = e?.id ?? `#${i}`;
    for (const p of validateEvent(e)) out.push(`${label}: ${p}`);
    if (ids.has(e?.id)) out.push(`${label}: duplicate id`);
    if (urls.has(e?.url)) out.push(`${label}: duplicate url`);
    ids.add(e?.id);
    urls.add(e?.url);
  });
  return out;
}
