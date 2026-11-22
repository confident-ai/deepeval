import events from "./events.json";

export type CommunityEvent = {
  id: string;
  title: string;
  /** ISO 8601 with offset, e.g. 2026-11-20T17:00:00-08:00 */
  start: string;
  /** IANA time zone of the venue, used to show local time */
  timezone: string;
  city: string;
  country: string;
  /** Luma event page */
  url: string;
  talk: string;
  host: string;
  hostGithub?: string;
  /** Cover from the Luma event, saved under public/community */
  image?: string;
  lat?: number;
  lng?: number;
  /** Run by Confident AI */
  official: boolean;
  /** GitHub issue the event was submitted in */
  issue?: number;
};

// An event moves to "past" this long after it starts. Events have no end
// time in the form, and most meetups run 2 to 3 hours.
const EVENT_LENGTH_MS = 4 * 60 * 60 * 1000;

export const communityEvents = events as CommunityEvent[];

export function splitEvents(now: number = Date.now()) {
  const withTime = communityEvents.map((e) => ({ e, t: Date.parse(e.start) }));
  const upcoming = withTime
    .filter(({ t }) => t + EVENT_LENGTH_MS > now)
    .sort((a, b) => a.t - b.t)
    .map(({ e }) => e);
  const past = withTime
    .filter(({ t }) => t + EVENT_LENGTH_MS <= now)
    .sort((a, b) => b.t - a.t)
    .map(({ e }) => e);
  return { upcoming, past };
}

export function countCities(list: CommunityEvent[]) {
  return new Set(
    list
      .filter((e) => e.city.toLowerCase() !== "online")
      .map((e) => `${e.city}|${e.country}`.toLowerCase()),
  ).size;
}

/** Date and time in the venue's own time zone, e.g. "Fri, Nov 20 · 5:00 PM PST". */
export function formatEventDate(e: CommunityEvent) {
  const d = new Date(e.start);
  const day = new Intl.DateTimeFormat("en-US", {
    timeZone: e.timezone,
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(d);
  const time = new Intl.DateTimeFormat("en-US", {
    timeZone: e.timezone,
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(d);
  return { day, time };
}

// Country name -> flag emoji, from the region names Intl knows, so a new
// country in events.json needs no lookup table here.
// Retired or grouping codes that Intl still names like a current country
// (DD is "Germany", FX "France", VD "Vietnam", ...); never map to them.
const RETIRED = new Set(["AN", "BU", "CS", "DD", "DY", "EU", "EZ", "FX", "HV", "NH", "NT", "QO", "RH", "SU", "TP", "UK", "UN", "VD", "XA", "XB", "YD", "YU", "ZR", "ZZ"]);

const REGION_BY_NAME = (() => {
  const names = new Intl.DisplayNames(["en"], { type: "region" });
  const map = new Map<string, string>();
  const A = 65;
  for (let i = 0; i < 26; i++) {
    for (let j = 0; j < 26; j++) {
      const code = String.fromCharCode(A + i, A + j);
      try {
        const name = names.of(code);
        if (name && name !== code && !RETIRED.has(code)) map.set(name.toLowerCase(), code);
      } catch {
        // Not a region code.
      }
    }
  }
  map.set("usa", "US").set("uk", "GB");
  return map;
})();

/** The ISO 3166-1 alpha-2 code for a country name, or undefined (e.g. "Online"). */
export const countryCode = (country: string) => REGION_BY_NAME.get(country.trim().toLowerCase());

/** The flag emoji for an alpha-2 code. */
export const flagOf = (code: string) =>
  String.fromCodePoint(...[...code.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65));

const REGION_NAMES = new Intl.DisplayNames(["en"], { type: "region" });
/** The English name for an alpha-2 code. */
export const countryName = (code: string) => REGION_NAMES.of(code) ?? code;

/** The flag emoji for a country name, or "" when it is not a known country (e.g. "Online"). */
export function countryFlag(country: string): string {
  const code = countryCode(country);
  return code ? flagOf(code) : "";
}
