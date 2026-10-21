import type { ChartColor } from "@/lib/charts/schema";

export interface Canvas {
  width: number;
  height: number;
  minWidth?: number;
  layoutWidth?: number;
}

export type Anchor = "start" | "middle" | "end";

export interface Scene {
  grid: { x1: number; y1: number; x2: number; y2: number }[];
  rings: string[];
  axis?: string;
  ticks: { x: number; y: number; text: string; fullText?: string; anchor: Anchor }[];

  axisLabels: { x: number; y: number; text: string; rotate?: boolean }[];
  categoryLabels: { x: number; y: number; lines: string[]; fullText?: string; rotate?: number; anchor: Anchor }[];
  thresholds: { d: string; label?: { x: number; y: number; text: string; anchor: Anchor } }[];

  barGroups: {
    i: number;
    from: "bottom" | "left";

    bars: { x: number; y: number; w: number; h: number; color: ChartColor; solid?: boolean }[];
  }[];

  polygons: { points: string; color: ChartColor; i: number; outline: boolean; opacity?: number; slice?: boolean }[];
  lines: { d: string; color: ChartColor; i: number; width: number }[];

  dots: { x: number; y: number; color: ChartColor; i: number; r?: number; bubble?: boolean }[];

  /** Matrix cells, each tinted with its own opacity. */
  cells: { x: number; y: number; w: number; h: number; color: ChartColor; opacity: number; i: number }[];

  labels: {
    x: number;
    y: number;
    text: string;
    color: ChartColor | "ink";
    anchor: Anchor;
    i: number;
    plain?: boolean;
    large?: boolean;
    /** 11px secondary text, e.g. a cell's share or name. */
    small?: boolean;
  }[];
}

export const emptyScene = (): Scene => ({
  grid: [], rings: [], ticks: [], axisLabels: [], categoryLabels: [], thresholds: [],
  barGroups: [], polygons: [], lines: [], dots: [], cells: [], labels: [],
});

export const linePath = (pts: (readonly [number, number])[], step = false) =>
  pts
    .map(([x, y], i) => {
      if (i === 0) return `M${x.toFixed(1)} ${y.toFixed(1)}`;
      return step ? `H${x.toFixed(1)}V${y.toFixed(1)}` : `L${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join("");

export const pointList = (pts: (readonly [number, number])[]) =>
  pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

export const textWidth = (text: string, size = 13) => text.length * size * 0.6;

export const assignLanes = (
  items: { x: number; width: number }[],
  left: number,
  right: number,
  gap = 12
): { lane: number; anchor: Anchor }[] => {
  const ends: number[] = [];
  const out: { lane: number; anchor: Anchor }[] = new Array(items.length);
  const order = items.map((item, i) => ({ ...item, i })).sort((a, b) => a.x - b.x);
  for (const item of order) {
    const anchor: Anchor = item.x - item.width / 2 < left ? "start" : item.x + item.width / 2 > right ? "end" : "middle";
    const from = anchor === "start" ? item.x : anchor === "end" ? item.x - item.width : item.x - item.width / 2;
    let lane = ends.findIndex(end => end + gap <= from);
    if (lane < 0) lane = ends.push(0) - 1;
    ends[lane] = from + item.width;
    out[item.i] = { lane, anchor };
  }
  return out;
};

