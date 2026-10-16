import { type Box, type Point, textWidth } from "@/components/frames/FlowPlot/_scene";
import {
  BOARD_SIDES,
  type BoardNote,
  type BoardNoteColumn,
  type BoardPullout,
  type BoardPulloutLine,
  type BoardData,
  type BoardRadius,
  type BoardSide,
  type BoardSize,
  type BoardType,
  pulloutLines,
  type LineEnd,
} from "@/lib/frames/board";
import { type Arranged, arrangeIcons } from "./_arrange";
import { layoutGroups } from "./_groups";
import {
  type Anchor,
  ARROW_LENGTH,
  type BoardScene,
  END_CIRCLE,
  inlineIcons,
  inlineWidth,
  MAGNIFIER,
  type SceneIcon,
  type ScenePullout,
  type SceneMark,
  type SceneText,
  TEXT_INSET,
} from "./_scene";
import { ROLE, TIER, weightOf } from "./_tokens";
import { layoutTree } from "./_tree";
import { layoutVenn } from "./_venn";

const PANEL_PAD = 32;
const HEADING_ROOM = 40;
const NOTE_GAP = 64;
const PULLOUT_GAP = 56;
const PULLOUT_STACK = 20;
const CARD_PAD = 16;
const PILL_GAP = 10;
const PILL_MIN = 120;
const BEND = 14;
const PLACE_GAP = 8;
const BOTTOM_GAP = 48;
const NOTE_SPREAD = 16;
const ARROW_LABEL_ROOM = 16;
const BLOCK_GAP = 6;
const PARALLEL = 20;
const LEVEL_RAISE = 0.45;
/** Room above and below the text of a note box, a pill and a card's heading. */
const NOTE_ROOM = 4;
const PILL_ROOM = 6;
const CARD_ROOM = 6;
const ANNOTATION = TIER[ROLE.annotation];
const PILL_H = ANNOTATION.line + PILL_ROOM;

const RADII: Record<BoardRadius, BoardScene["radius"]> = {
  none: { panel: 0, box: 0, mark: 0, card: 0 },
  small: { panel: 6, box: 8, mark: 2, card: 4 },
  medium: { panel: 14, box: 16, mark: 4, card: 10 },
  large: { panel: 22, box: 24, mark: 10, card: 16 },
};

const END_ROOM: Record<LineEnd, number> = { magnifier: MAGNIFIER, circle: END_CIRCLE, dot: 0, arrow: ARROW_LENGTH - 1, none: 0 };

interface Content {
  w: number;
  h: number;
  icons: Arranged | null;
  tier: BoardSize;
}

const blockOf = (refs: string[], arrange: BoardNote["arrange"], fallback: "line" | "grid", seed: string, tier: BoardSize) =>
  refs.length ? arrangeIcons(refs.length, arrange ?? fallback, seed, TIER[tier].icon) : null;

const placeBlock = (refs: string[], block: Arranged | null, cx: number, top: number, tier: BoardSize): SceneIcon[] => {
  const size = TIER[tier].icon;
  return block ? refs.map((ref, i) => ({ ref, x: cx - block.w / 2 + block.points[i].x - size / 2, y: top + block.points[i].y - size / 2, w: size, h: size })) : [];
};

const inlineMark = (box: Omit<SceneMark, "text">, label: string, refs: string[], tier: BoardSize) => {
  const inline = inlineWidth(refs.length, tier);
  const cy = box.y + box.h / 2;
  const start = box.x + (box.w - inline - textWidth(label, TIER[tier].text)) / 2;
  return {
    mark: { ...box, text: refs.length ? { x: start + inline, y: cy, text: label, anchor: "start" as const, tier } : { x: box.x + box.w / 2, y: cy, text: label, anchor: "middle" as const, tier } },
    icons: inlineIcons(refs, start, cy, tier),
  };
};

const noteContent = (note: BoardNote): Content => {
  const tier = note.size ?? ROLE.label;
  const { text, line } = TIER[tier];
  if (note.shape === "box") return { w: inlineWidth(note.icons.length, tier) + textWidth(note.label, text) + 2 * TEXT_INSET, h: line + NOTE_ROOM, icons: null, tier };
  const icons = blockOf(note.icons, note.arrange, "line", note.label, tier);
  return { w: Math.max(textWidth(note.label, text), icons?.w ?? 0), h: (icons ? icons.h + BLOCK_GAP : 0) + line, icons, tier };
};

const placeNote = (note: BoardNote, size: Content, cx: number, cy: number) => {
  const box = { x: cx - size.w / 2, y: cy - size.h / 2, w: size.w, h: size.h, color: note.color };
  if (note.shape === "box") {
    const { mark, icons } = inlineMark(box, note.label, note.icons, size.tier);
    return { note: { ...mark, bare: false }, icons };
  }
  const textY = box.y + (size.icons ? size.icons.h + BLOCK_GAP : 0) + TIER[size.tier].line / 2;
  return {
    note: { ...box, bare: true, text: { x: cx, y: textY, text: note.label, anchor: "middle" as const, tier: size.tier } },
    icons: placeBlock(note.icons, size.icons, cx, box.y, size.tier),
  };
};

const ends = (arrow: BoardNote["arrow"]) => ({ start: arrow === "in" || arrow === "both", end: arrow === "out" || arrow === "both" });

const columnSize = (column?: BoardNoteColumn) => {
  if (!column) return { w: 0, gap: 0, sizes: [] as Content[] };
  const sizes = column.items.map(noteContent);
  const labels = column.items.map(n => (n.arrowLabel ? textWidth(n.arrowLabel, ANNOTATION.text) + 2 * ARROW_LABEL_ROOM : 0));
  return {
    w: Math.max(...sizes.map(s => s.w), column.heading ? textWidth(column.heading, TIER[ROLE.title].text) : 0),
    gap: Math.max(NOTE_GAP, ...labels),
    sizes,
  };
};

const pulloutSize = (pullout: BoardPullout) => {
  const bare = pullout.shape === "none";
  const tier = pullout.size ?? (bare ? ROLE.actor : ROLE.label);
  const { text, line } = TIER[tier];
  const cardHead = line + CARD_ROOM;
  const icons = blockOf(pullout.icons, pullout.arrange, "grid", pullout.for ?? pullout.heading ?? "pullout", tier);
  const pills = pullout.items.map(i => inlineWidth(i.icons.length, ROLE.annotation) + textWidth(i.label, ANNOTATION.text) + 2 * TEXT_INSET);
  const heading = pullout.heading ? textWidth(pullout.heading, text) : 0;
  const inner = Math.max(pullout.items.length ? PILL_MIN : 0, ...pills, icons?.w ?? 0, heading);
  const pillsH = pullout.items.length ? pullout.items.length * PILL_H + (pullout.items.length - 1) * PILL_GAP : 0;
  if (bare) {
    const above = (icons ? icons.h + BLOCK_GAP : 0) + (pullout.heading ? line : 0);
    const h = above + (pillsH ? (above ? PILL_GAP : 0) + pillsH : 0);
    return { bare, tier, cardHead, inner, icons, w: inner, h, half: { w: inner / 2, h: h / 2 }, r: 0 };
  }
  const w = inner + 2 * CARD_PAD;
  const h = 2 * CARD_PAD + (pullout.heading ? cardHead : 0) + (icons?.h ?? 0) + (icons && pillsH ? PILL_GAP : 0) + pillsH;
  const r = Math.hypot(w / 2, h / 2) + 10;
  return { bare, tier, cardHead, inner, icons, w, h, half: { w: r, h: r }, r };
};

const rectRim = (center: Point, half: { w: number; h: number }, from: Point): Point => {
  const [dx, dy] = [from.x - center.x, from.y - center.y];
  const t = Math.min(dx ? half.w / Math.abs(dx) : Infinity, dy ? half.h / Math.abs(dy) : Infinity);
  return { x: center.x + dx * t, y: center.y + dy * t };
};

const lineLabel = (points: Point[], text: string): SceneText => {
  const segments = points.slice(1).map((p, i) => [points[i], p] as const);
  const [a, b] = segments.reduce((best, seg) => (Math.hypot(seg[1].x - seg[0].x, seg[1].y - seg[0].y) > Math.hypot(best[1].x - best[0].x, best[1].y - best[0].y) ? seg : best));
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  return Math.abs(b.x - a.x) >= Math.abs(b.y - a.y)
    ? { x: mid.x, y: mid.y - 9, text, anchor: "middle", tier: ROLE.annotation }
    : { x: mid.x + 8, y: mid.y, text, anchor: "start", tier: ROLE.annotation };
};

const unit = (from: Point, to: Point): Point => {
  const d = Math.hypot(to.x - from.x, to.y - from.y) || 1;
  return { x: (to.x - from.x) / d, y: (to.y - from.y) / d };
};

const toward = (from: Point, to: Point, by: number): Point => {
  const u = unit(from, to);
  return { x: from.x + u.x * by, y: from.y + u.y * by };
};

const labelled = (text: string, refs: string[], x: number, y: number, anchor: "start" | "middle") => {
  const tier = ROLE.annotation;
  if (!refs.length) return { text: { x, y, text, anchor, tier } as SceneText, icons: [] as SceneIcon[] };
  const inline = inlineWidth(refs.length, tier);
  const left = anchor === "middle" ? x - (inline + textWidth(text, ANNOTATION.text)) / 2 : x;
  return { text: { x: left + inline, y, text, anchor: "start", tier } as SceneText, icons: inlineIcons(refs, left, y, tier) };
};

const labelWidth = (line: BoardPulloutLine) =>
  line.label ? inlineWidth(line.icons.length, ROLE.annotation) + textWidth(line.label, ANNOTATION.text) + 2 * ARROW_LABEL_ROOM : 0;

const sceneLines = (pullout: BoardPullout, path: Point[], edge: number) => {
  const lines = pulloutLines(pullout);
  const icons: SceneIcon[] = [];
  const u = unit(path[0], path[1]);
  const normal = Math.abs(u.x) >= Math.abs(u.y) ? { x: 0, y: 1 } : { x: 1, y: 0 };
  const scene = lines.map((line, k) => {
    const offset = (k - (lines.length - 1) / 2) * PARALLEL;
    const shifted = path.map(p => ({ x: p.x + normal.x * offset, y: p.y + normal.y * offset }));
    const [first, second] = shifted;
    const [beforeLast, last] = shifted.slice(-2);
    let label: SceneText | undefined;
    if (line.label) {
      const level = Math.abs(last.y - beforeLast.y) < Math.abs(last.x - beforeLast.x);
      const below = lines.length > 1 && k === lines.length - 1;
      const spot = level ? { x: (edge + last.x) / 2, y: last.y + (below ? 11 : -9), anchor: "middle" as const } : lineLabel(shifted, line.label);
      const placed = labelled(line.label, line.icons, spot.x, spot.y, spot.anchor === "start" ? "start" : "middle");
      label = placed.text;
      icons.push(...placed.icons);
    }
    const points = [...shifted];
    points[0] = toward(first, second, END_ROOM[line.start]);
    points[points.length - 1] = toward(last, beforeLast, END_ROOM[line.end]);
    return {
      points,
      style: line.style,
      color: line.color,
      start: { shape: line.start, at: first, dir: unit(second, first) },
      end: { shape: line.end, at: last, dir: unit(beforeLast, last) },
      label,
    };
  });
  return { lines: scene, icons };
};

export const layoutBoard = (data: BoardData, type: BoardType): BoardScene => {
  const pulloutSizes = { left: (data.pullouts.left ?? []).map(pulloutSize), right: (data.pullouts.right ?? []).map(pulloutSize) };
  const pulloutGap = (side: BoardSide) =>
    Math.max(PULLOUT_GAP, ...(data.pullouts[side] ?? []).flatMap(c => pulloutLines(c).map(labelWidth)));
  const gaps = { left: pulloutGap("left"), right: pulloutGap("right") };
  const reach = (side: BoardSide) => (pulloutSizes[side].length ? Math.max(...pulloutSizes[side].map(s => 2 * s.half.w)) + gaps[side] : 0);
  const columns = { left: columnSize(data.notes.left), right: columnSize(data.notes.right) };

  const layoutMain = (x: number, y: number) =>
    type === "groups" ? layoutGroups(data.main, x, y) : type === "venn" ? layoutVenn(data.main, x, y) : layoutTree(data.main.items, x, y);
  const columnSpan = (column: (typeof columns)["left"]) => (column.w ? column.w + column.gap : 0);
  const content = columnSpan(columns.left) + layoutMain(0, 0).w + columnSpan(columns.right);
  const widen = data.heading ? Math.max(0, textWidth(data.heading.text, TIER[ROLE.heading].text) - content) / 2 : 0;

  const panelX = reach("left");
  const leftX = panelX + PANEL_PAD + widen;
  const treeX = leftX + columnSpan(columns.left);
  const treeY = PANEL_PAD + (data.heading ? HEADING_ROOM : 0);
  const main = layoutMain(treeX, treeY);
  const treeRight = treeX + main.w;
  const treeBottom = treeY + main.h;
  const rightX = treeRight + columns.right.gap;
  const drawn = [...main.boxes, ...main.bars];
  const edges =
    data.main.border !== "none" || !drawn.length
      ? { left: treeX, right: treeRight, top: treeY, bottom: treeBottom }
      : {
          left: Math.min(...drawn.map(b => b.x)),
          right: Math.max(...drawn.map(b => b.x + b.w)),
          top: Math.min(...drawn.map(b => b.y)),
          bottom: Math.max(...drawn.map(b => b.y + b.h)),
        };
  const whole: Anchor = {
    x: (edges.left + edges.right) / 2,
    y: (edges.top + edges.bottom) / 2,
    left: edges.left,
    right: edges.right,
    gutter: treeBottom + PANEL_PAD / 2,
    root: -1,
    box: { x: edges.left, y: edges.top, w: edges.right - edges.left, h: edges.bottom - edges.top },
  };
  const anchor = (id?: string) => (id ? main.anchors.get(id)! : whole);

  const notes: BoardScene["notes"] = [];
  const icons: SceneIcon[] = [];
  const arrows: BoardScene["arrows"] = [];
  const arrowLabels: SceneText[] = [];
  const columnHeadings: SceneText[] = [];
  for (const side of BOARD_SIDES) {
    const column = data.notes[side];
    if (!column) continue;
    const x = side === "left" ? leftX : rightX;
    const { w, sizes } = columns[side];
    if (column.heading) columnHeadings.push({ x: x + w / 2, y: treeY, text: column.heading, anchor: "middle", tier: ROLE.title, weight: weightOf("title") });
    column.items.forEach((item, i) => {
      const a = anchor(item.for);
      const placed = placeNote(item, sizes[i], x + w / 2, a.y);
      notes.push(placed.note);
      icons.push(...placed.icons);
      const [from, to] = side === "right" ? [a.right, placed.note.x - 1] : [a.left, placed.note.x + placed.note.w + 1];
      arrows.push({ d: `M${from},${a.y}H${to}`, ...ends(item.arrow) });
      if (item.arrowLabel) arrowLabels.push({ x: (from + to) / 2, y: a.y - 9, text: item.arrowLabel, anchor: "middle", tier: ROLE.annotation });
    });
  }

  let panelRight = (columns.right.w ? rightX + columns.right.w : treeRight) + PANEL_PAD + widen;
  let panelBottom = treeBottom + PANEL_PAD;
  const bottom = data.notes.bottom?.items ?? [];
  if (bottom.length) {
    const top = treeBottom + BOTTOM_GAP;
    const placed = bottom
      .map(item => ({ item, size: noteContent(item), cx: anchor(item.for).x }))
      .sort((p, q) => p.cx - q.cx);
    let edge = leftX - NOTE_SPREAD;
    for (const p of placed) {
      p.cx = Math.max(p.cx, edge + NOTE_SPREAD + p.size.w / 2);
      edge = p.cx + p.size.w / 2;
      const { note, icons: noteIcons } = placeNote(p.item, p.size, p.cx, top + p.size.h / 2);
      notes.push(note);
      icons.push(...noteIcons);
      const from = p.item.for && data.main.border === "none" ? anchor(p.item.for).box.y + anchor(p.item.for).box.h : edges.bottom;
      arrows.push({ d: `M${p.cx},${from}V${top - 1}`, ...ends(p.item.arrow) });
      if (p.item.arrowLabel) arrowLabels.push({ x: p.cx + 8, y: (from + top) / 2, text: p.item.arrowLabel, anchor: "start", tier: ROLE.annotation });
    }
    panelRight = Math.max(panelRight, edge + PANEL_PAD);
    panelBottom = Math.max(...placed.map(p => top + p.size.h)) + PANEL_PAD;
  }
  const panel = { x: panelX, y: 0, w: panelRight - panelX, h: panelBottom, border: data.panel };

  const pullouts: ScenePullout[] = [];
  for (const side of BOARD_SIDES) {
    const dir = side === "right" ? 1 : -1;
    let floor = -Infinity;
    (data.pullouts[side] ?? []).forEach((pullout, i) => {
      const size = pulloutSizes[side][i];
      const target = anchor(pullout.for);
      const start = { x: side === "right" ? edges.right : edges.left, y: target.y };
      const gap = gaps[side];
      const circle = { x: side === "right" ? panelRight + gap + size.half.w : panelX - gap - size.half.w, y: 0, r: size.r };
      const level = size.bare ? start.y : start.y - size.r * LEVEL_RAISE;
      const wanted = { above: start.y - size.half.h - PLACE_GAP, level, below: start.y + size.half.h + PLACE_GAP }[pullout.place];
      circle.y = Math.max(wanted, floor + PULLOUT_STACK + size.half.h);
      floor = circle.y + size.half.h;
      const bend = { x: start.x + dir * BEND, y: target.gutter };
      const exit = { x: side === "right" ? panelRight + gap / 2 : panelX - gap / 2, y: bend.y };
      const via = pullout.route === "gap" ? [bend, exit] : [];
      const from = via.at(-1) ?? start;
      const rim = size.bare ? rectRim(circle, { w: size.half.w + BLOCK_GAP, h: size.half.h + BLOCK_GAP }, from) : toward(circle, from, size.r);
      const card = { x: circle.x - size.w / 2, y: circle.y - size.h / 2, w: size.w, h: size.h };
      let top = size.bare ? card.y : card.y + CARD_PAD + (pullout.heading ? size.cardHead : 0);
      icons.push(...placeBlock(pullout.icons, size.icons, circle.x, top, size.tier));
      if (size.icons) top += size.icons.h + (size.bare ? BLOCK_GAP : PILL_GAP);
      let heading: SceneText | undefined;
      if (pullout.heading) {
        const line = TIER[size.tier].line;
        heading = size.bare
          ? { x: circle.x, y: top + line / 2, text: pullout.heading, anchor: "middle", tier: size.tier, weight: weightOf("actor") }
          : { x: circle.x, y: card.y + CARD_PAD + size.cardHead / 2 - 4, text: pullout.heading, anchor: "middle", tier: size.tier };
        if (size.bare) top += line + PILL_GAP;
      }
      const pills = pullout.items.map((item, j) => {
        const placed = inlineMark({ x: circle.x - size.inner / 2, y: top + j * (PILL_H + PILL_GAP), w: size.inner, h: PILL_H, color: item.color }, item.label, item.icons, ROLE.annotation);
        icons.push(...placed.icons);
        return placed.mark;
      });
      const drawn = sceneLines(pullout, [start, ...via, rim], side === "right" ? panelRight : panelX);
      icons.push(...drawn.icons);
      pullouts.push({
        shape: pullout.shape,
        lines: drawn.lines,
        circle,
        extent: size.half,
        card,
        heading,
        pills,
      });
    });
  }

  const heading = data.heading
    ? {
        x: data.heading.align === "left" ? leftX : data.heading.align === "right" ? panelRight - PANEL_PAD : panelX + panel.w / 2,
        y: PANEL_PAD + 8,
        text: data.heading.text,
        tier: ROLE.heading,
        weight: weightOf("heading"),
        anchor: data.heading.align === "left" ? ("start" as const) : data.heading.align === "right" ? ("end" as const) : ("middle" as const),
      }
    : undefined;

  const viewTop = Math.min(0, ...pullouts.map(l => l.circle.y - l.extent.h - MAGNIFIER)) - 2;
  const viewBottom = Math.max(panel.h, ...pullouts.map(l => l.circle.y + l.extent.h + MAGNIFIER)) + 2;
  const viewRight = Math.max(panelRight, ...pullouts.map(l => l.circle.x + l.extent.w + MAGNIFIER)) + 2;
  const viewLeft = Math.min(0, ...pullouts.map(l => l.circle.x - l.extent.w - MAGNIFIER)) - 2;

  const colors = [...main.bars, ...main.boxes, ...notes, ...pullouts.flatMap(l => l.pills)].flatMap(m => (m.color ? [m.color] : []));
  const hatch = data.fill === "hatch" ? [...new Set(colors)] : [];

  return {
    view: { x: viewLeft, y: viewTop, w: viewRight - viewLeft, h: viewBottom - viewTop },
    radius: RADII[data.radius],
    hatch,
    panel,
    heading,
    mainBox: data.main.border === "none" ? undefined : { x: treeX, y: treeY, w: main.w, h: main.h, border: data.main.border },
    main,
    arrows,
    arrowLabels,
    columnHeadings,
    notes,
    pullouts,
    icons,
  };
};
