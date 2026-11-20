#!/usr/bin/env node
/**
 * Adds one approved community event to lib/community/events.json.
 *
 * Run by .github/workflows/community-event.yml when a maintainer adds the
 * `event-approved` label to a `community-event` issue. The issue body is
 * untrusted input: it only arrives through the ISSUE_BODY env var, every
 * field is validated, and nothing from it reaches a shell.
 *
 * It also saves the Luma cover to public/community/<id>.jpg, so the site
 * never loads images from Luma. A missing cover is not an error; the reason
 * goes to the listing PR.
 *
 * Env: ISSUE_BODY, ISSUE_NUMBER, ISSUE_AUTHOR, GITHUB_OUTPUT (optional).
 * Run from docs/. Exits 1 with reasons on stderr if the issue is not valid,
 * and 2 on an internal error.
 */
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { EVENTS_FILE, isCalendarDate, isTimeZone, validateEvent, validateEvents } from "../../lib/community/event-schema.mjs";

// Labels in .github/ISSUE_TEMPLATE/community-event.yml. Keep them in sync.
const FIELDS = {
  title: "Event title",
  url: "Luma link",
  date: "Date",
  time: "Start time",
  timezone: "Time zone",
  city: "City",
  country: "Country",
  talk: "DeepEval talk",
  host: "Organizer name",
};
const REQUIRED_CHECKS = [
  "co-host",
  "opt-in",
  "free",
  "Code of Conduct",
];

/** Issue forms render as "### Label\n\nvalue" blocks. */
export function parseIssueForm(body) {
  const out = {};
  const parts = body.replace(/\r\n/g, "\n").split(/^### /m).slice(1);
  for (const part of parts) {
    const nl = part.indexOf("\n");
    const label = part.slice(0, nl).trim();
    const value = part.slice(nl + 1).trim();
    // The free-text "Anything else" field sits after every input and before
    // "Confirmations", which is the last block of the form. So the first
    // block wins for inputs, and the last block wins for Confirmations: a
    // "### Label" typed in the free text cannot replace either.
    if (label in out && label !== "Confirmations") continue;
    out[label] = value === "_No response_" ? "" : value;
  }
  return out;
}

/** Offset in minutes of `timeZone` at the UTC instant `ms`. */
function zoneOffset(ms, timeZone) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone,
      hourCycle: "h23",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
      .formatToParts(new Date(ms))
      .map((p) => [p.type, p.value]),
  );
  const local = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute, +parts.second);
  return Math.round((local - ms) / 60000);
}

/** "2026-11-20" + "17:00" in America/Los_Angeles → "2026-11-20T17:00:00-08:00". */
export function toIsoWithOffset(date, time, timeZone) {
  const [y, m, d] = date.split("-").map(Number);
  const [h, min] = time.split(":").map(Number);
  const wall = Date.UTC(y, m - 1, d, h, min);
  let offset = zoneOffset(wall, timeZone);
  offset = zoneOffset(wall - offset * 60000, timeZone);
  const sign = offset < 0 ? "-" : "+";
  const abs = Math.abs(offset);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  return `${date}T${time}:00${sign}${hh}:${mm}`;
}

const slug = (s) =>
  s
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40)
    .replace(/-+$/g, "");

export function eventFromIssue(body, issueNumber, author) {
  const form = parseIssueForm(body);
  const problems = [];
  const get = (key) => (form[FIELDS[key]] ?? "").trim();

  for (const key of Object.keys(FIELDS)) if (!get(key)) problems.push(`"${FIELDS[key]}" is empty`);

  const checks = form["Confirmations"] ?? "";
  for (const word of REQUIRED_CHECKS) {
    const line = checks.split("\n").find((l) => l.includes(word));
    if (!line || !/^- \[[xX]\]/.test(line.trim())) problems.push(`the "${word}" confirmation is not checked`);
  }

  const date = get("date");
  const time = get("time");
  const timezone = get("timezone");
  if (date && !isCalendarDate(date)) problems.push("Date must be a real date in YYYY-MM-DD");
  if (time && !/^([01]\d|2[0-3]):[0-5]\d$/.test(time)) problems.push("Start time must be HH:MM in 24-hour time");
  if (timezone && !isTimeZone(timezone)) problems.push("Time zone must be an IANA name, e.g. Europe/London");
  if (problems.length) return { problems };

  const city = get("city");
  const event = {
    id: `${date}-${slug(city) || "online"}`,
    title: get("title"),
    start: toIsoWithOffset(date, time, timezone),
    timezone,
    city,
    country: get("country"),
    url: get("url").split(/[?#]/)[0].replace(/\/+$/, ""),
    talk: get("talk"),
    host: get("host"),
    hostGithub: author,
    official: false,
    issue: Number(issueNumber),
  };
  return { event, problems: validateEvent(event) };
}

const COVER_URL = /"cover_url":"(https:\/\/images\.lumacdn\.com\/[A-Za-z0-9/._-]+\.(?:png|jpe?g|webp|gif))"/;
const MAX_COVER_BYTES = 2 * 1024 * 1024;

/**
 * Saves the Luma cover as public/community/<id>.jpg.
 * Returns { image } on success, or { reason } so the listing PR can say why
 * the cover is missing (a Luma page change would otherwise stop every cover
 * without a signal).
 */
export async function saveCover(eventUrl, id) {
  try {
    const page = await fetch(eventUrl, { headers: { "user-agent": "deepeval-community-bot" }, signal: AbortSignal.timeout(15000) });
    if (!page.ok) return { reason: `the Luma page returned HTTP ${page.status}` };
    const match = (await page.text()).match(COVER_URL);
    if (!match) return { reason: "no cover_url on the Luma page" };
    // Luma's CDN resizes the upload to 640px wide and returns a JPEG.
    const resized = match[1].replace("https://images.lumacdn.com/", "https://images.lumacdn.com/cdn-cgi/image/format=jpeg,fit=cover,quality=80,width=640/");
    const img = await fetch(resized, { signal: AbortSignal.timeout(15000) });
    const type = img.headers.get("content-type") ?? "";
    if (!img.ok) return { reason: `the cover returned HTTP ${img.status}` };
    if (!type.startsWith("image/jpeg")) return { reason: `the cover has content-type ${type || "none"}` };
    const bytes = Buffer.from(await img.arrayBuffer());
    if (bytes.length === 0 || bytes.length > MAX_COVER_BYTES) return { reason: `the cover is ${bytes.length} bytes` };
    // JPEG files start with FF D8 FF.
    if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes[2] !== 0xff) return { reason: "the cover is not a JPEG file" };
    fs.mkdirSync(path.join("public", "community"), { recursive: true });
    fs.writeFileSync(path.join("public", "community", `${id}.jpg`), bytes);
    return { image: `/community/${id}.jpg` };
  } catch (err) {
    return { reason: `error: ${err?.message ?? err}` };
  }
}

async function main() {
  const { ISSUE_BODY = "", ISSUE_NUMBER = "", ISSUE_AUTHOR = "" } = process.env;
  const { event, problems } = eventFromIssue(ISSUE_BODY, ISSUE_NUMBER, ISSUE_AUTHOR);
  const fail = (list) => {
    console.error(list.map((p) => `- ${p}`).join("\n"));
    process.exit(1);
  };
  if (problems.length) fail(problems);

  const list = JSON.parse(fs.readFileSync(EVENTS_FILE, "utf8"));
  if (list.some((e) => e.url === event.url)) fail([`${event.url} is already listed`]);
  let id = event.id;
  for (let n = 2; list.some((e) => e.id === id); n++) id = `${event.id}-${n}`;
  event.id = id;

  const cover = await saveCover(event.url, event.id);
  const coverNote = cover.image
    ? "Cover: saved from Luma."
    : `Cover: not saved (${cover.reason}). The card shows the first two letters of the city.`;
  if (cover.image) event.image = cover.image;
  console.log(coverNote);

  list.push(event);
  list.sort((a, b) => Date.parse(a.start) - Date.parse(b.start));
  const all = validateEvents(list);
  if (all.length) fail(all);

  fs.writeFileSync(EVENTS_FILE, `${JSON.stringify(list, null, 2)}\n`);
  if (process.env.GITHUB_OUTPUT) {
    // One line each: the reason text comes from fetch errors, never from the issue.
    fs.appendFileSync(process.env.GITHUB_OUTPUT, `id=${event.id}\ncover=${coverNote.replace(/[\r\n]+/g, " ")}\n`);
  }
  console.log(`Added ${event.id}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main();
  } catch (err) {
    // Exit 2 = a bug or an infrastructure problem, not bad input. The
    // workflow reports it as an automation error, not to the organizer.
    console.error(`internal error: ${err?.message ?? err}`);
    process.exit(2);
  }
}
