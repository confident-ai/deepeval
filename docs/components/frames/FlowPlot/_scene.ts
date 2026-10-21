import type { ChartColor } from "@/lib/charts/schema";
import type { FlowIcon, NodeShape } from "@/lib/frames/flow";

// The geometry FlowPlot draws: boxes, containers and lines, positioned by a
// frame's own layout (FlowFrame's dagre layout, SchemaFrame's rows). Text sizes
// match the charts: 13px labels, 11px small print.

export const LABEL = 13;
export const SMALL = 11;
export const LINE = 17;
export const SMALL_LINE = 14;
// Average glyph width of the inherited proportional font, as the charts estimate it.
const CHAR = 0.6;

export const textWidth = (text: string, size = LABEL) => Array.from(text).length * size * CHAR;

/**
 * Wrap at word boundaries without dropping text. A word too long for a line
 * breaks after its hyphens ("Reference-" / "less"), and only then mid-word.
 */
export const wrapWords = (text: string, width: number, size = LABEL): string[] => {
  const max = Math.max(4, Math.floor(width / (size * CHAR)));
  // Pieces, each with whether it follows the previous one after a space.
  const pieces = text
    .trim()
    .split(/\s+/)
    .flatMap(word =>
      Array.from(word).length > max ? word.split(/(?<=-)/).map((part, i) => ({ part, spaced: i === 0 })) : [{ part: word, spaced: true }]
    );
  const lines: string[] = [];
  let line = "";
  for (const { part, spaced } of pieces) {
    const joined = line ? `${line}${spaced ? " " : ""}${part}` : part;
    if (Array.from(joined).length <= max) {
      line = joined;
      continue;
    }
    if (line) lines.push(line);
    const chars = Array.from(part);
    while (chars.length > max) lines.push(chars.splice(0, max).join(""));
    line = chars.join("");
  }
  if (line) lines.push(line);
  return lines;
};

export interface Point {
  x: number;
  y: number;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SceneNode extends Box {
  id: string;
  lines: string[];
  color: ChartColor;
  shape: NodeShape;
  /** Drawn left of the label, which is centered in the space after it. */
  icon?: FlowIcon;
  /** Smaller secondary lines under the label, e.g. an example value. */
  sub?: string[];
  /** A dashed outline, e.g. for an optional field. */
  dashed?: boolean;
  /** An empty dashed slot marking something absent. */
  ghost?: boolean;
}

export interface SceneGroup extends Box {
  id: string;
  label: string;
  badge?: { text: string; w: number; color?: ChartColor };
}

export interface SceneEdge {
  d: string;
  dashed: boolean;
  both: boolean;
  label?: { x: number; y: number; lines: string[] };
  /** A colored connector that ends in a dot instead of an arrowhead. */
  color?: ChartColor;
}

export interface FlowScene {
  width: number;
  height: number;
  nodes: SceneNode[];
  groups: SceneGroup[];
  edges: SceneEdge[];
  /** Sequence diagrams only. */
  lifelines: { x: number; y1: number; y2: number }[];
}

export const PAD_X = 14;
export const PAD_Y = 9;
// Icons sit left of the label at the label's cap height.
export const ICON = 15;
const ICON_GAP = 7;
/** Horizontal room an icon takes in a node, or 0. */
export const iconRoom = (icon?: FlowIcon) => (icon ? ICON + ICON_GAP : 0);

// How far a drawing may shrink to fit before it scrolls instead, so labels stay
// legible (13px at 0.75 is about the size of a phone chart's tick labels).
export const MIN_SCALE = { wide: 0.8, narrow: 0.75 };

export const GROUP_HEAD = 22;
export const GROUP_PAD = 12;

/** Width a container needs for its header: the uppercase, letter-spaced label and its badge. */
export const groupHeaderWidth = (label: string, badgeW = 0) =>
  // 0.08em tracking and uppercase glyphs run about 30% wider than the estimate.
  textWidth(label.toUpperCase(), SMALL) * 1.3 + 20 + (badgeW ? badgeW + 10 : 0);

/** Height of a label block: label lines plus any smaller lines under them. */
export const blockHeight = (lines: number, sub = 0) => lines * LINE + (sub ? 2 + sub * SMALL_LINE : 0);

/** Where the line from a node's center towards `p` leaves the node's outline. */
export const boundary = (n: SceneNode, p: Point): Point => {
  const cx = n.x + n.w / 2;
  const cy = n.y + n.h / 2;
  const dx = p.x - cx;
  const dy = p.y - cy;
  const len = Math.hypot(dx, dy);
  if (!len) return p;
  const ux = dx / len;
  const uy = dy / len;
  let t: number;
  if (n.shape === "circle") t = n.w / 2;
  else if (n.shape === "diamond") t = 1 / (Math.abs(ux) / (n.w / 2) + Math.abs(uy) / (n.h / 2));
  else t = Math.min(ux ? n.w / 2 / Math.abs(ux) : Infinity, uy ? n.h / 2 / Math.abs(uy) : Infinity);
  return { x: cx + ux * t, y: cy + uy * t };
};

const f = (n: number) => Math.round(n * 10) / 10;

/** A smooth path through layout points: straight ends, rounded bends. */
export const smoothPath = (pts: Point[]): string => {
  if (pts.length < 3) return `M${f(pts[0].x)},${f(pts[0].y)}L${f(pts.at(-1)!.x)},${f(pts.at(-1)!.y)}`;
  const mid = (a: Point, b: Point) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  let d = `M${f(pts[0].x)},${f(pts[0].y)}`;
  const first = mid(pts[0], pts[1]);
  d += `L${f(first.x)},${f(first.y)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const m = mid(pts[i], pts[i + 1]);
    d += `Q${f(pts[i].x)},${f(pts[i].y)} ${f(m.x)},${f(m.y)}`;
  }
  const last = pts.at(-1)!;
  return `${d}L${f(last.x)},${f(last.y)}`;
};

/** A horizontal S-curve from `a` to `b`, leaving and entering level. */
export const sCurve = (a: Point, b: Point): string => {
  const dx = (b.x - a.x) / 2;
  return `M${f(a.x)},${f(a.y)}C${f(a.x + dx)},${f(a.y)} ${f(b.x - dx)},${f(b.y)} ${f(b.x)},${f(b.y)}`;
};

/** Shift everything so the drawing starts at a small margin, and measure it. */
export const normalize = (scene: FlowScene): FlowScene => {
  const boxes: Box[] = [...scene.nodes, ...scene.groups];
  for (const e of scene.edges) {
    if (e.label) {
      const w = Math.max(...e.label.lines.map(l => textWidth(l, SMALL)));
      const h = e.label.lines.length * SMALL_LINE;
      boxes.push({ x: e.label.x - w / 2, y: e.label.y - h / 2, w, h });
    }
  }
  const nums = (d: string) => (d.match(/-?\d+(\.\d+)?/g) ?? []).map(Number);
  for (const e of scene.edges) {
    const n = nums(e.d);
    for (let i = 0; i + 1 < n.length; i += 2) boxes.push({ x: n[i], y: n[i + 1], w: 0, h: 0 });
  }
  const M = 4;
  const minX = Math.min(...boxes.map(b => b.x)) - M;
  const minY = Math.min(...boxes.map(b => b.y)) - M;
  const maxX = Math.max(...boxes.map(b => b.x + b.w)) + M;
  const maxY = Math.max(...boxes.map(b => b.y + b.h)) + M;
  const shift = <T extends Box>(b: T): T => ({ ...b, x: b.x - minX, y: b.y - minY });
  const shiftPath = (d: string) => {
    let i = 0;
    return d.replace(/-?\d+(\.\d+)?/g, m => String(Math.round((Number(m) - (i++ % 2 === 0 ? minX : minY)) * 10) / 10));
  };
  return {
    width: Math.ceil(maxX - minX),
    height: Math.ceil(maxY - minY),
    nodes: scene.nodes.map(shift),
    groups: scene.groups.map(shift),
    edges: scene.edges.map(e => ({
      ...e,
      d: shiftPath(e.d),
      label: e.label ? { ...e.label, x: e.label.x - minX, y: e.label.y - minY } : undefined,
    })),
    lifelines: scene.lifelines.map(l => ({ x: l.x - minX, y1: l.y1 - minY, y2: l.y2 - minY })),
  };
};
