import type { Point } from "@/components/frames/FlowPlot/_scene";
import type { IconArrangement } from "@/lib/frames/board";
import { TIER } from "./_tokens";

/** The icon size the spacing below is tuned for; other sizes scale it. */
const BASE = TIER.sm.icon;
const PITCH = 40;
const ROW_PITCH = 36;
const ZIGZAG = 14;
const SCATTER_PITCH = 46;
const SCATTER_JITTER = 16;

export interface Arranged {
  w: number;
  h: number;
  points: Point[];
}

const seeded = (seed: string) => {
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return () => {
    h = (h + 0x6d2b79f5) | 0;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const fit = (points: Point[], size: number): Arranged => {
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const [minX, minY] = [Math.min(...xs), Math.min(...ys)];
  return {
    w: Math.max(...xs) - minX + size,
    h: Math.max(...ys) - minY + size,
    points: points.map(p => ({ x: p.x - minX + size / 2, y: p.y - minY + size / 2 })),
  };
};

const grid = (n: number, cols: number, pitch: number, rowPitch: number): Point[] =>
  Array.from({ length: n }, (_, i) => {
    const row = Math.floor(i / cols);
    const inRow = Math.min(cols, n - row * cols);
    return { x: ((cols - inRow) * pitch) / 2 + (i % cols) * pitch, y: row * rowPitch };
  });

export const arrangeIcons = (n: number, arrange: IconArrangement, seed: string, size: number): Arranged => {
  const placed = (points: Point[]) => fit(points.map(p => ({ x: (p.x * size) / BASE, y: (p.y * size) / BASE })), size);
  if (n === 1) return placed([{ x: 0, y: 0 }]);
  if (arrange === "line") return placed(Array.from({ length: n }, (_, i) => ({ x: i * PITCH, y: 0 })));
  if (arrange === "zigzag-h") return placed(Array.from({ length: n }, (_, i) => ({ x: i * PITCH * 0.8, y: i % 2 ? 0 : ZIGZAG })));
  if (arrange === "zigzag-v") return placed(Array.from({ length: n }, (_, i) => ({ x: i % 2 ? ZIGZAG * 1.4 : 0, y: i * ROW_PITCH * 0.8 })));
  if (arrange === "circle") {
    const r = Math.max(PITCH * 0.75, (PITCH * n) / (2 * Math.PI));
    return placed(Array.from({ length: n }, (_, i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
      return { x: r * Math.cos(a), y: r * Math.sin(a) };
    }));
  }
  if (arrange === "scatter") {
    const random = seeded(seed);
    const cols = Math.ceil(Math.sqrt(n));
    return placed(
      grid(n, cols, SCATTER_PITCH, SCATTER_PITCH).map(p => ({
        x: p.x + (random() - 0.5) * SCATTER_JITTER,
        y: p.y + (random() - 0.5) * SCATTER_JITTER,
      }))
    );
  }
  return placed(grid(n, n <= 3 ? n : Math.ceil(Math.sqrt(n)), PITCH, ROW_PITCH));
};
