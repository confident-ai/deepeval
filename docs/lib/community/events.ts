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
