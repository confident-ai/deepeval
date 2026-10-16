import type { Box, Point } from "@/components/frames/FlowPlot/_scene";
import type { ChartColor } from "@/lib/charts/schema";
import type { BoardPullout, BoardPulloutLine, BoardItem, BoardSize, LineEnd } from "@/lib/frames/board";
import { INLINE_GAP, TIER } from "./_tokens";

export const TEXT_INSET = 12;
export const MAIN_PAD = 24;
export const MAGNIFIER = 9;
export const END_CIRCLE = 5;
export const END_DOT = 3.5;
export const ARROW_LENGTH = 8;
export const MIN_SCALE = 0.6;

export interface SceneText {
  x: number;
  y: number;
  text: string;
  anchor: "start" | "middle" | "end";
  tier: BoardSize;
  weight?: "medium";
}

export interface SceneMark extends Box {
  color?: ChartColor;
  text: SceneText;
}

export interface SceneIcon extends Box {
  ref: string;
}

export interface SceneBox extends Box {
  id: string;
  color?: ChartColor;
  border?: BoardItem["border"];
  shape?: "circle" | "ellipse";
  depth: number;
  /** An overlapping venn set: a see-through wash, so the overlap shows. */
  wash?: boolean;
  /** Holds boxes of its own (`split: "boxes"`): drawn with the larger corner radius. */
  holds?: boolean;
}

export interface SceneEnd {
  shape: LineEnd;
  at: Point;
  dir: Point;
}

export interface ScenePulloutLine {
  points: Point[];
  style: BoardPulloutLine["style"];
  color?: ChartColor;
  start: SceneEnd;
  end: SceneEnd;
  label?: SceneText;
}

export interface ScenePullout {
  shape: BoardPullout["shape"];
  lines: ScenePulloutLine[];
  circle: Point & { r: number };
  extent: { w: number; h: number };
  card: Box;
  heading?: SceneText;
  pills: SceneMark[];
}

export interface Anchor {
  x: number;
  y: number;
  left: number;
  right: number;
  gutter: number;
  root: number;
  box: Box;
}

export interface SceneRule {
  d: string;
  color?: ChartColor;
  border?: BoardItem["border"];
}

/** Text that's only an ellipsis: "⋮" (vertical), "…", "⋯" or "..." (horizontal). */
const ELLIPSIS_TEXT = /^\s*(⋮|…|⋯|\.{3})\s*$/;

export const ellipsisOf = (text: string): { vertical: boolean } | null => (ELLIPSIS_TEXT.test(text) ? { vertical: text.trim() === "⋮" } : null);

export interface SceneDots extends Point {
  vertical: boolean;
}

export interface MainLayout {
  w: number;
  h: number;
  anchors: Map<string, Anchor>;
  bars: (SceneMark & { id: string })[];
  elbows: string[];
  boxes: SceneBox[];
  rules: SceneRule[];
  labels: (SceneText & { role: "title" | "label" | "body" })[];
  icons: SceneIcon[];
  links: { d: string; start: boolean; end: boolean; dashed: boolean }[];
  linkLabels: SceneText[];
  dots: SceneDots[];
}

export const emptyMain = (w: number, h: number): MainLayout => ({
  w,
  h,
  anchors: new Map(),
  bars: [],
  elbows: [],
  boxes: [],
  rules: [],
  labels: [],
  icons: [],
  links: [],
  linkLabels: [],
  dots: [],
});

/** The room inline icons take in front of text of a tier, gap included. */
export const inlineWidth = (count: number, tier: BoardSize) => (count ? count * TIER[tier].inline + (count - 1) * INLINE_GAP + TIER[tier].gap : 0);

export const inlineIcons = (refs: string[], x: number, cy: number, tier: BoardSize): SceneIcon[] => {
  const size = TIER[tier].inline;
  return refs.map((ref, i) => ({ ref, x: x + i * (size + INLINE_GAP), y: cy - size / 2, w: size, h: size }));
};

export interface BoardScene {
  view: Box;
  radius: { panel: number; box: number; mark: number; card: number };
  hatch: ChartColor[];
  panel: Box & { border: "solid" | "none" };
  heading?: SceneText;
  mainBox?: Box & { border: "solid" | "dashed" };
  main: MainLayout;
  arrows: { d: string; start: boolean; end: boolean }[];
  arrowLabels: SceneText[];
  columnHeadings: SceneText[];
  notes: (SceneMark & { bare: boolean })[];
  pullouts: ScenePullout[];
  icons: SceneIcon[];
}
