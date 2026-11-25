"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { Star } from "lucide-react";
import { useInView } from "@site/src/components/Mascot/_useInView";
import {
  countryCode,
  flagOf,
  splitEvents,
  type CommunityEvent,
} from "@/lib/community/events";
import centers from "@/lib/community/country-centers.json";
import names from "@/lib/community/country-names.json";
import contributorCountries from "@/lib/generated/contributor-countries.json";
import stargazerCountries from "@/lib/generated/stargazer-countries.json";
import { LAND_BITS, LAND_H, LAND_W } from "./_land";
import styles from "./MeetupGlobe.module.scss";

// A planet rising from the bottom of the hero, drawn on the mascot's pixel
// grid. Flags stand on poles planted in the countries that host meetups.
const GW = 192;
const R = 82;
const CX = GW / 2;
const CY = R + 6;
// The whole disc is drawn; the pane's bottom edge cuts it off wherever the
// space runs out, so it always meets the bottom like a horizon.
const GH = CY + R;
// Tilted so the band shows the northern half, where the flags stand.
const TILT = (-16 * Math.PI) / 180;
const SPIN_PER_TICK = 0.6; // degrees
const TICK_MS = 90;
const MAX_FLAGS = 45;
const START_LNG = 12; // degrees east: central Europe faces the viewer first
const MIN_APART = 3; // degrees of arc; only near-identical spots are merged

const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

const land = (() => {
  const bin =
    typeof atob === "function"
      ? atob(LAND_BITS)
      : Buffer.from(LAND_BITS, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
})();

const isLand = (lat: number, lon: number) => {
  const y = clamp(Math.floor(90 - lat), 0, LAND_H - 1);
  const x = ((Math.floor(lon + 180) % LAND_W) + LAND_W) % LAND_W;
  const i = y * LAND_W + x;
  return ((land[i >> 3] >> (i & 7)) & 1) === 1;
};

/** World lat/lng (degrees) -> view coords (x right, y up, z toward the viewer). */
function toView(lat: number, lng: number, spin: number) {
  const la = rad(lat);
  const lo = rad(lng - spin);
  const x = Math.cos(la) * Math.sin(lo);
  const y1 = Math.sin(la);
  const z1 = Math.cos(la) * Math.cos(lo);
  return {
    x,
    y: y1 * Math.cos(TILT) - z1 * Math.sin(TILT),
    z: y1 * Math.sin(TILT) + z1 * Math.cos(TILT),
  };
}

type Shade = "land3" | "land2" | "land1" | "sea";

/** One frame of the planet as run-length paths, one per shade. */
function drawPlanet(spin: number): Record<Shade, string> {
  const out: Record<Shade, string> = { land3: "", land2: "", land1: "", sea: "" };
  const ct = Math.cos(TILT);
  const st = Math.sin(TILT);
  for (let row = 0; row < GH; row++) {
    const v = (CY - (row + 0.5)) / R;
    let key: Shade | null = null;
    let start = 0;
    for (let col = 0; col <= GW; col++) {
      let next: Shade | null = null;
      if (col < GW) {
        const u = (col + 0.5 - CX) / R;
        const d = u * u + v * v;
        if (d <= 1) {
          const z = Math.sqrt(1 - d);
          const y1 = v * ct + z * st;
          const z1 = -v * st + z * ct;
          const lat = deg(Math.asin(clamp(y1, -1, 1)));
          const lng = deg(Math.atan2(u, z1)) + spin;
          if (isLand(lat, lng)) next = z > 0.6 ? "land3" : z > 0.3 ? "land2" : "land1";
          else if ((row + col) % 2 === 0) next = "sea";
        }
      }
      if (next !== key) {
        if (key) out[key] += `M${start} ${row}h${col - start}v1h-${col - start}z`;
        key = next;
        start = col;
      }
    }
  }
  return out;
}

type Spot = {
  key: string;
  code: string;
  lat: number;
  lng: number;
  events: CommunityEvent[];
  contributors?: { count: number; logins: string[] };
  stars?: number;
};

const arc = (a: Spot, b: Spot) => {
  const p = toView(a.lat, a.lng, 0);
  const q = toView(b.lat, b.lng, 0);
  return deg(Math.acos(clamp(p.x * q.x + p.y * q.y + p.z * q.z, -1, 1)));
};

const CENTERS = centers as Record<string, number[]>;
// A fixed table, not Intl: Node and browsers name some regions
// differently ("Hong Kong SAR China"), which would break hydration.
const countryName = (code: string) => (names as Record<string, string>)[code] ?? code;
const CONTRIBUTORS = new Map(
  (contributorCountries as { code: string; count: number; logins: string[] }[]).map((c) => [c.code, c])
);
const STARS = new Map((stargazerCountries as { code: string; count: number }[]).map((s) => [s.code, s.count]));
// A contributor counts for more than a star when ranking countries.
const weight = (code: string) => (CONTRIBUTORS.get(code)?.count ?? 0) * 20 + (STARS.get(code) ?? 0);

/**
 * One flag per country. Countries with meetups come first (most meetups
 * first, placed at the mean of their cities), then countries ranked by
 * the contributors and stargazers who list them on GitHub, skipping any
 * too close to a flag already placed.
 */
function pickSpots(): Spot[] {
  const { upcoming, past } = splitEvents();
  const byCode = new Map<string, CommunityEvent[]>();
  for (const e of [...upcoming, ...past]) {
    const code = countryCode(e.country);
    if (!code || typeof e.lat !== "number" || typeof e.lng !== "number") continue;
    byCode.set(code, [...(byCode.get(code) ?? []), e]);
  }
  const real = [...byCode.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .map<Spot>(([code, events]) => ({
      key: code,
      code,
      lat: events.reduce((s, e) => s + e.lat!, 0) / events.length,
      lng: events.reduce((s, e) => s + e.lng!, 0) / events.length,
      events,
      contributors: CONTRIBUTORS.get(code),
      stars: STARS.get(code),
    }));
  const open = [...new Set([...CONTRIBUTORS.keys(), ...STARS.keys()])]
    .filter((code) => !byCode.has(code) && CENTERS[code])
    .sort((a, b) => weight(b) - weight(a))
    .map<Spot>((code) => ({
      key: code,
      code,
      lat: CENTERS[code][0],
      lng: CENTERS[code][1],
      events: [],
      contributors: CONTRIBUTORS.get(code),
      stars: STARS.get(code),
    }));

  const placed: Spot[] = [];
  for (const s of [...real, ...open]) {
    if (placed.length >= MAX_FLAGS) break;
    if (placed.every((p) => arc(p, s) >= MIN_APART)) placed.push(s);
  }
  return placed;
}

const MeetupGlobe: React.FC = () => {
  const { ref, inView } = useInView({ rootMargin: "100px" });
  // Start centered on Europe, where most of the flags are.
  const spots = useMemo(pickSpots, []);
  const [spin, setSpin] = useState(START_LNG);
  const [still, setStill] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  const drag = useRef<{ x: number; spin: number; id: number; perDeg: number } | null>(null);
  // Spin left over from a drag, in degrees per tick; it eases back to the
  // normal pace so the planet keeps turning the moment it is let go.
  const momentum = useRef(0);
  const lastMove = useRef<{ spin: number; t: number } | null>(null);

  useEffect(() => {
    setStill(window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }, []);

  useEffect(() => {
    if (!inView || still) return;
    const id = window.setInterval(() => {
      if (drag.current || hovered) return;
      const extra = momentum.current;
      momentum.current = Math.abs(extra) < 0.05 ? 0 : extra * 0.9;
      setSpin((s) => s + SPIN_PER_TICK + extra);
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [inView, still, hovered]);

  // Redraw the map in whole-degree steps, so it moves pixel by pixel.
  const mapSpin = Math.round(spin);
  const planet = useMemo(() => drawPlanet(mapSpin), [mapSpin]);

  // Drag sideways to spin: the planet follows the pointer at its center.
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    const width = e.currentTarget.getBoundingClientRect().width;
    const radiusPx = (R / GW) * width;
    drag.current = { x: e.clientX, spin, id: e.pointerId, perDeg: rad(1) * radiusPx };
    momentum.current = 0;
    lastMove.current = { spin, t: performance.now() };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const next = d.spin - (e.clientX - d.x) / d.perDeg;
    const now = performance.now();
    const last = lastMove.current;
    if (last && now > last.t) {
      // Degrees per tick at the pointer's current speed, capped.
      const v = ((next - last.spin) / (now - last.t)) * TICK_MS;
      momentum.current = clamp(v - SPIN_PER_TICK, -12, 12);
    }
    lastMove.current = { spin: next, t: now };
    setSpin(next);
  };
  const onPointerUp = () => {
    drag.current = null;
    // A drag that ended on a still pointer leaves no fling.
    if (lastMove.current && performance.now() - lastMove.current.t > 120) momentum.current = 0;
  };

  const placed = spots.map((spot) => {
    const p = toView(spot.lat, spot.lng, spin);
    return {
      spot,
      x: CX + p.x * R,
      y: CY - p.y * R,
      z: p.z,
      visible: p.z > 0.08 && CY - p.y * R < GH - 2,
    };
  });
  const tip = placed.find((q) => q.spot.key === hovered && q.visible);

  return (
    <div ref={ref} className={styles.globe}>
      <div
        className={styles.stage}
        style={{ aspectRatio: `${GW} / ${GH}` }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <svg
          className={styles.planet}
          viewBox={`0 0 ${GW} ${GH}`}
          shapeRendering="crispEdges"
          role="img"
          aria-label="A pixel planet turning, with flags planted in the countries that host DeepEval meetups. Drag to spin it."
        >
          <path className={styles.sea} d={planet.sea} />
          <path className={styles.land1} d={planet.land1} />
          <path className={styles.land2} d={planet.land2} />
          <path className={styles.land3} d={planet.land3} />
        </svg>

        {placed.map(({ spot, x, y, z, visible }) => {
          const real = spot.events.length > 0;
          // Farther flags stand a little shorter, so the planet reads as round.
          const scale = (0.7 + 0.3 * clamp(z, 0, 1)).toFixed(2);
          return (
            <button
              key={spot.key}
              type="button"
              className={styles.pin}
              data-real={real || undefined}
              data-hidden={!visible || undefined}
              tabIndex={visible ? 0 : -1}
              aria-label={`${countryName(spot.code)}: ${
                real
                  ? `${spot.events.length} DeepEval meetup${spot.events.length > 1 ? "s" : ""}`
                  : "no meetup yet"
              }${spot.contributors ? `, ${spot.contributors.count} contributors` : ""}${
                spot.stars ? `, ${spot.stars} stars` : ""
              }`}
              style={{
                left: `${((x / GW) * 100).toFixed(2)}%`,
                top: `${((y / GH) * 100).toFixed(2)}%`,
                zIndex: Math.max(1, Math.round(z * 100)),
                transform: `translate(-1px, -100%) scale(${scale})`,
              }}
              onPointerEnter={() => setHovered(spot.key)}
              onPointerLeave={() => setHovered((h) => (h === spot.key ? null : h))}
              onFocus={() => setHovered(spot.key)}
              onBlur={() => setHovered((h) => (h === spot.key ? null : h))}
            >
              <span className={styles.flag}>{flagOf(spot.code)}</span>
              {spot.events.length > 1 && <span className={styles.count}>{spot.events.length}</span>}
            </button>
          );
        })}

        {tip && (
          <div
            className={styles.tip}
            style={{
              left: `${clamp((tip.x / GW) * 100, 18, 82).toFixed(2)}%`,
              top: `${((tip.y / GH) * 100).toFixed(2)}%`,
            }}
            role="status"
          >
            <strong className={styles.tipTitle}>
              {flagOf(tip.spot.code)} {countryName(tip.spot.code)}
            </strong>
            <span className={styles.tipStats}>
              {tip.spot.stars && (
                <span title="GitHub stars">
                  <Star size={11} strokeWidth={2} aria-label="GitHub stars" />
                  {tip.spot.stars.toLocaleString("en-US")}
                </span>
              )}
              {tip.spot.contributors && (
                <span>
                  {tip.spot.contributors.count} contributor
                  {tip.spot.contributors.count > 1 ? "s" : ""}
                </span>
              )}
              <span>
                {tip.spot.events.length > 0
                  ? `${tip.spot.events.length} meetup${tip.spot.events.length > 1 ? "s" : ""}`
                  : "no meetup yet"}
              </span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default MeetupGlobe;
