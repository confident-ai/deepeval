import { type Box, textWidth, wrapWords } from "@/components/frames/FlowPlot/_scene";
import type { BoardData, BoardItem, BoardLink, BoardSize, Direction, LabelPlace } from "@/lib/frames/board";
import { type Arranged, arrangeIcons } from "./_arrange";
import { type Anchor, ellipsisOf, emptyMain, inlineIcons, inlineWidth, MAIN_PAD, type MainLayout, type SceneRule } from "./_scene";
import { ROLE, TIER, weightOf } from "./_tokens";

/** Room around a title, on top of its line: a box's head, or a cell's and circle's label. */
const HEAD_ROOM = 12;
const CELL_ROOM = 8;
const PAD = 14;
const BOX_GAP = 18;
const LINKED_GAP = 40;
const LANE_START = 14;
const LANE = 12;
const MIN_BOX = 140;
const MIN_CELL = 96;
const MIN_CIRCLE = 56;
const LABEL_ROOM = 16;
const PAIR_SPREAD = 18;
const TEXT_WRAP = 240;
const CONTENT_GAP = 8;
const BODY = TIER[ROLE.body];

interface Row {
  text: string;
  icons: string[];
  indent: number;
  centered: boolean;
}

interface TextBlock {
  w: number;
  h: number;
  rows: Row[];
}

/** Items laid out in rows of `cols`, every column as wide as the widest item, every row as tall as its tallest. */
interface Grid {
  cols: number;
  cellW: number;
  rows: number[];
  w: number;
  h: number;
}

const gridOf = (items: { w: number; h: number }[], columns: number, gap: number): Grid => {
  const cols = Math.min(columns, items.length);
  const cellW = Math.max(...items.map(i => i.w));
  const rows = Array.from({ length: Math.ceil(items.length / cols) }, (_, r) => Math.max(...items.slice(r * cols, r * cols + cols).map(i => i.h)));
  return { cols, cellW, rows, w: cols * cellW + (cols - 1) * gap, h: rows.reduce((a, b) => a + b, 0) + (rows.length - 1) * gap };
};

/** Places items into a grid stretched to w × h; `rule` draws the lines between cells (none between boxes). */
const placeGrid = <T,>(items: T[], grid: Grid, x: number, y: number, w: number, h: number, gap: number, put: (item: T, x: number, y: number, w: number, h: number) => void, rule?: (d: string) => void) => {
  const cellW = (w - gap * (grid.cols - 1)) / grid.cols;
  const extra = (h - grid.h) / grid.rows.length;
  let top = y;
  grid.rows.forEach((rowH, r) => {
    const height = rowH + extra;
    if (r && rule) rule(`M${x},${top}H${x + w}`);
    items.slice(r * grid.cols, r * grid.cols + grid.cols).forEach((item, c) => {
      const left = x + c * (cellW + gap);
      if (c && rule) rule(`M${left},${top}V${top + height}`);
      put(item, left, top, cellW, height);
    });
    top += height + gap;
  });
};

interface Sized {
  item: BoardItem;
  depth: number;
  dir: Direction;
  boxed: boolean;
  at: LabelPlace;
  band: number;
  titleTier: BoardSize;
  titleRole: "title" | "label";
  titleIcons: string[];
  /** The title, wrapped like body text when it's long. */
  titleLines: string[];
  gap: number;
  iconPx: number;
  icons: Arranged | null;
  text: TextBlock | null;
  content: { w: number; h: number };
  kids: Sized[];
  grid: Grid | null;
  diameter: number;
  w: number;
  h: number;
}

type Frame = Omit<SceneRule, "d">;

const across = (dir: Direction): Direction => (dir === "vertical" ? "horizontal" : "vertical");
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

const textBlock = (lines: BoardItem["lines"]): TextBlock | null => {
  if (!lines.length) return null;
  const rows = lines.flatMap(line => {
    const indent = inlineWidth(line.icons.length, ROLE.body);
    return wrapWords(line.text, TEXT_WRAP - indent, BODY.text).map((text, i) => ({ text, icons: i ? [] : line.icons, indent, centered: !line.icons.length }));
  });
  return { w: Math.max(...rows.map(r => r.indent + textWidth(r.text, BODY.text))), h: rows.length * BODY.line, rows };
};

const labelW = (link: BoardLink) => (link.label ? textWidth(link.label, TIER[ROLE.annotation].text) : 0);

const pad = (links: BoardLink[], route: BoardLink["route"]) => {
  const on = links.filter(l => l.route === route);
  const label = Math.max(0, ...on.map(labelW));
  const room = !label ? 0 : route === "left" || route === "right" ? label + 8 : 14;
  return Math.max(MAIN_PAD, LANE_START + Math.max(0, on.length - 1) * LANE + 12 + room);
};

export const layoutGroups = (main: BoardData["main"], x0: number, y0: number): MainLayout => {
  const dir = main.direction ?? "vertical";
  const parentOf = new Map<string, BoardItem>();
  const walk = (items: BoardItem[]) =>
    items.forEach(item => {
      item.children.forEach(child => parentOf.set(child.id, item));
      walk(item.children);
    });
  walk(main.items);
  const siblings = (link: BoardLink) => {
    const parent = parentOf.get(link.from);
    return parent !== undefined && parent === parentOf.get(link.to) && parent.split === "boxes";
  };
  const direct = main.links.filter(l => l.route === "direct");
  const gapFor = (links: BoardLink[], along: Direction) =>
    links.length ? Math.max(LINKED_GAP, along === "horizontal" ? Math.max(0, ...links.map(labelW)) + 2 * LABEL_ROOM : 0) : BOX_GAP;

  const measure = (item: BoardItem, depth: number, parentDir: Direction, boxed: boolean): Sized => {
    const itemDir = item.direction ?? across(parentDir);
    const split = item.split === "boxes";
    const circle = item.shape === "circle";
    const headed = boxed && !circle;
    const titleRole = headed ? "title" : "label";
    const titleTier = item.size ?? ROLE[titleRole];
    const iconPx = TIER[item.size ?? ROLE.icon].icon;
    const titleIcons = item.iconsAt === "label" ? item.icons : [];
    const icons = item.icons.length && !titleIcons.length ? arrangeIcons(item.icons.length, item.arrange ?? "grid", item.id, iconPx) : null;
    const text = textBlock(item.lines);
    const content = {
      w: Math.max(icons?.w ?? 0, text?.w ?? 0),
      h: (icons?.h ?? 0) + (icons && text ? CONTENT_GAP : 0) + (text?.h ?? 0),
    };
    const kids = item.children.map(child => measure(child, depth + 1, itemDir, split));
    const ids = new Set(item.children.map(c => c.id));
    const gap = split ? gapFor(direct.filter(l => siblings(l) && ids.has(l.from)), itemDir) : 0;
    const at = item.labelAt ?? (circle ? "below" : "head");
    const grid = item.columns && kids.length ? gridOf(kids, item.columns, split ? BOX_GAP : 0) : null;
    const spread = (values: number[]) => sum(values) + gap * (values.length - 1);
    const inset = split && item.border !== "none" ? 2 * PAD : 0;
    const kidsW = !kids.length ? 0 : (grid ? grid.w : itemDir === "horizontal" ? spread(kids.map(k => k.w)) : Math.max(...kids.map(k => k.w))) + inset;
    const kidsH = !kids.length ? 0 : (grid ? grid.h : itemDir === "horizontal" ? Math.max(...kids.map(k => k.h)) : spread(kids.map(k => k.h))) + inset;
    const title = TIER[titleTier];
    // A title wraps only when it's wider than both the body-text width and what its box already holds.
    const titleIconsW = inlineWidth(titleIcons.length, titleTier);
    const titleLines = wrapWords(item.label, Math.max(TEXT_WRAP, content.w, kidsW - 2 * PAD) - titleIconsW, title.text);
    const label = Math.max(...titleLines.map(line => textWidth(line, title.text))) + titleIconsW + 2 * PAD;
    const band = Math.max(titleLines.length * title.line, titleIcons.length ? title.inline : 0) + (headed ? HEAD_ROOM : CELL_ROOM);
    const base = { item, depth, dir: itemDir, boxed, at, band, titleTier, titleRole, titleIcons, titleLines, gap, iconPx, icons, text, content, kids, grid, diameter: 0 } as const;
    if (circle) {
      const diameter = Math.max(MIN_CIRCLE, Math.hypot(content.w, content.h) + PAD);
      return { ...base, diameter, w: Math.max(label, diameter), h: band + diameter };
    }
    if (at === "below") return { ...base, w: Math.max(label, content.w + 2 * PAD), h: PAD + content.h + band };
    const contentH = content.h ? content.h + (boxed ? 2 * PAD : PAD) : 0;
    return {
      ...base,
      w: Math.max(boxed ? MIN_BOX : MIN_CELL, label, content.w + 2 * PAD, kidsW),
      h: band + contentH + kidsH,
    };
  };

  const sized = main.items.map(item => measure(item, 0, dir, true));
  const gap = gapFor(direct.filter(l => !siblings(l)), dir);
  const along = (s: Sized) => (dir === "vertical" ? s.h : s.w);
  const cross = Math.max(...sized.map(s => (dir === "vertical" ? s.w : s.h)));
  const length = sum(sized.map(along)) + (sized.length - 1) * gap;
  const [padL, padR, padT, padB] = (["left", "right", "top", "bottom"] as const).map(side => pad(main.links, side));
  const mainGrid = main.columns ? gridOf(sized, main.columns, BOX_GAP) : null;
  const layout = emptyMain(
    (mainGrid ? mainGrid.w : dir === "vertical" ? cross : length) + padL + padR,
    (mainGrid ? mainGrid.h : dir === "vertical" ? length : cross) + padT + padB
  );
  const childDir = new Map<string, Direction>();

  const placeContent = (s: Sized, cx: number, top: number) => {
    if (s.icons) {
      const { w, points } = s.icons;
      s.item.icons.forEach((ref, i) => {
        const p = points[i];
        layout.icons.push({ ref, x: cx - w / 2 + p.x - s.iconPx / 2, y: top + p.y - s.iconPx / 2, w: s.iconPx, h: s.iconPx });
      });
    }
    if (!s.text) return;
    const textTop = top + (s.icons ? s.icons.h + CONTENT_GAP : 0);
    const left = cx - s.text.w / 2;
    s.text.rows.forEach((row, i) => {
      const y = textTop + i * BODY.line + BODY.line / 2;
      const text = { y, text: row.text, tier: ROLE.body, role: "body" } as const;
      const dots = row.centered && ellipsisOf(row.text);
      if (dots) {
        layout.dots.push({ x: cx, y, ...dots });
        return;
      }
      if (row.centered) {
        layout.labels.push({ ...text, x: cx, anchor: "middle" });
        return;
      }
      layout.icons.push(...inlineIcons(row.icons, left, y, ROLE.body));
      layout.labels.push({ ...text, x: left + row.indent, anchor: "start" });
    });
  };

  /** The title's lines centered on (cx, cy), its icons in front; with `left`, starting at that x instead (a section heading). */
  const placeTitle = (s: Sized, cx: number, cy: number, left?: number, width?: number) => {
    const dots = ellipsisOf(s.item.label);
    if (dots) {
      layout.dots.push({ x: cx, y: cy, ...dots });
      return;
    }
    const { text: size, line } = TIER[s.titleTier];
    const iconsW = inlineWidth(s.titleIcons.length, s.titleTier);
    // A box stretched wider than it measured can take its title on fewer lines.
    const lines = width && s.titleLines.length > 1 ? wrapWords(s.item.label, width - iconsW, size) : s.titleLines;
    const start = left ?? (iconsW ? cx - (iconsW + Math.max(...lines.map(l => textWidth(l, size)))) / 2 : undefined);
    const firstY = cy - ((lines.length - 1) / 2) * line;
    if (iconsW) layout.icons.push(...inlineIcons(s.titleIcons, start!, firstY, s.titleTier));
    lines.forEach((text, i) => {
      const y = cy + (i - (lines.length - 1) / 2) * line;
      const label = { y, text, tier: s.titleTier, role: s.titleRole, weight: weightOf(s.titleRole) };
      layout.labels.push(start === undefined ? { ...label, x: cx, anchor: "middle" } : { ...label, x: start + iconsW, anchor: "start" });
    });
  };

  /** A box that's only a title keeps its own height in a row instead of stretching to its neighbours'. */
  const compact = (s: Sized) => s.at === "head" && !s.kids.length && !s.content.h && !s.diameter;

  const place = (s: Sized, x: number, y: number, w: number, h: number, anchor: Omit<Anchor, "x" | "y" | "box">, frame: Frame = {}) => {
    const { id, color, border } = s.item;
    if (s.diameter) {
      const cx = x + w / 2;
      const centered = y + Math.max(0, (h - s.diameter) / 2);
      const top = s.at === "above" ? Math.max(centered, y + s.band) : Math.min(centered, y + h - s.h);
      const body = { x: cx - s.diameter / 2, y: top, w: s.diameter, h: s.diameter };
      layout.anchors.set(id, { ...anchor, x: cx, y: body.y + s.diameter / 2, box: body });
      placeTitle(s, cx, s.at === "above" ? body.y - s.band / 2 : body.y + s.diameter + s.band / 2 + 1);
      layout.boxes.push({ id, ...body, color, border, shape: "circle", depth: s.depth });
      placeContent(s, cx, body.y + (s.diameter - s.content.h) / 2);
      return;
    }
    const below = s.at === "below" ? y + (h - s.content.h - s.band) / 2 + (s.content.h ? PAD / 2 : 0) : 0;
    // A borderless item that's only a title is anchored on its title, so arrows meet it head-on.
    const bareTitle = s.at === "below" && !s.content.h && border === "none";
    const box = s.at === "above" ? { x, y: y + s.band, w, h: h - s.band } : bareTitle ? { x, y: below, w, h: s.band } : { x, y, w, h };
    layout.anchors.set(id, { ...anchor, x: x + w / 2, y: box.y + box.h / 2, box });
    const framed = s.boxed && border !== "none";
    if (framed) layout.boxes.push({ id, ...box, color, border, depth: s.depth, holds: s.item.split === "boxes" && s.kids.length > 0 });
    const inner = framed ? { color, border } : frame;
    const rule = (d: string) => layout.rules.push({ d, ...inner });
    if (s.at === "below") {
      placeContent(s, x + w / 2, below);
      placeTitle(s, x + w / 2, below + s.content.h + s.band / 2, undefined, w - 2 * PAD);
      return;
    }
    const titleOnly = !s.content.h && !s.kids.length;
    // A borderless row's title above it reads as a section heading: flush left. A column of cards keeps it centered.
    const heading = s.at === "above" && border === "none" && s.kids.length > 0 && s.dir === "horizontal";
    placeTitle(s, x + w / 2, s.at === "head" && titleOnly ? y + h / 2 : y + s.band / 2 + (s.boxed ? 0 : 3), heading ? x : undefined, w - 2 * PAD);
    if (s.at === "head" && framed && (s.content.h || s.kids.length)) rule(`M${x},${y + s.band}H${x + w}`);
    let top = y + s.band;
    const split = s.item.split === "boxes";
    if (s.content.h) {
      const area = s.kids.length ? s.content.h + (s.boxed ? 2 * PAD : PAD) : y + h - top;
      const blockTop = s.kids.length ? top + (s.boxed ? PAD : 0) : top + (area - s.content.h) / 2 - (s.boxed ? 0 : PAD / 4);
      placeContent(s, x + w / 2, blockTop);
      top += area;
      if (s.kids.length && !split) rule(`M${x},${top}H${x + w}`);
    }
    if (!s.kids.length) return;
    childDir.set(id, s.dir);
    const bottom = y + h;
    if (s.grid) {
      const inset = split && border !== "none" ? PAD : 0;
      const put = (kid: Sized, kx: number, ky: number, kw: number, kh: number) => place(kid, kx, ky, kw, kh, anchor, inner);
      placeGrid(s.kids, s.grid, x + inset, top + inset, w - 2 * inset, bottom - top - 2 * inset, split ? BOX_GAP : 0, put, split ? undefined : rule);
      return;
    }
    if (split) {
      const inset = border === "none" ? 0 : PAD;
      const [ix, iy, iw, ih] = [x + inset, top + inset, w - 2 * inset, bottom - top - 2 * inset];
      if (s.dir === "horizontal") {
        const extra = (iw - sum(s.kids.map(k => k.w)) - s.gap * (s.kids.length - 1)) / s.kids.length;
        let cx = ix;
        for (const kid of s.kids) {
          if (compact(kid)) place(kid, cx, iy + (ih - kid.h) / 2, kid.w + extra, kid.h, anchor, inner);
          else place(kid, cx, iy, kid.w + extra, ih, anchor, inner);
          cx += kid.w + extra + s.gap;
        }
      } else {
        let cy = iy;
        for (const kid of s.kids) {
          place(kid, ix, cy, iw, kid.h, anchor, inner);
          cy += kid.h + s.gap;
        }
      }
      return;
    }
    if (s.dir === "horizontal") {
      const extra = (w - sum(s.kids.map(k => k.w))) / s.kids.length;
      let cx = x;
      s.kids.forEach((kid, i) => {
        if (i) rule(`M${cx},${top}V${bottom}`);
        place(kid, cx, top, kid.w + extra, bottom - top, anchor, inner);
        cx += kid.w + extra;
      });
    } else {
      const extra = (bottom - top - sum(s.kids.map(k => k.h))) / s.kids.length;
      let cy = top;
      s.kids.forEach((kid, i) => {
        if (i) rule(`M${x},${cy}H${x + w}`);
        place(kid, x, cy, w, kid.h + extra, anchor, inner);
        cy += kid.h + extra;
      });
    }
  };

  const roots: Box[] = [];
  if (mainGrid) {
    placeGrid(sized, mainGrid, x0 + padL, y0 + padT, mainGrid.w, mainGrid.h, BOX_GAP, (s, x, y, w, h) => {
      place(s, x, y, w, h, { left: x - 2, right: x + w + 2, gutter: y + h + BOX_GAP / 2, root: roots.length });
      roots.push({ x, y, w, h });
    });
    return layout;
  }
  // Plain boxes beside a bordered group line up with the boxes inside it.
  const groupInset = sized.some(s => s.item.split === "boxes" && s.kids.length && s.item.border !== "none") ? PAD : 0;
  let cursor = dir === "vertical" ? y0 + padT : x0 + padL;
  sized.forEach((s, i) => {
    const [x, y] = dir === "vertical" ? [x0 + padL, cursor] : [cursor, y0 + padT];
    const [w, h] = dir === "vertical" ? [cross, s.h] : [s.w, cross];
    const last = i === sized.length - 1;
    const gutter = dir === "vertical" && !last ? y + h + gap / 2 : y + h + padB / 2;
    const inset = !s.kids.length && !s.diameter ? groupInset : 0;
    const [px, py, pw, ph] =
      dir === "vertical" ? [x + inset, y, w - 2 * inset, h] : compact(s) ? [x, y + (h - s.h) / 2, w, s.h] : [x, y + inset, w, h - 2 * inset];
    place(s, px, py, pw, ph, { left: px - 2, right: px + pw + 2, gutter, root: i });
    roots.push(s.at === "above" ? layout.anchors.get(s.item.id)!.box : { x, y, w, h });
    cursor += along(s) + gap;
  });

  const used = { left: 0, right: 0, top: 0, bottom: 0 };
  const pairs = new Map<string, BoardLink[]>();
  for (const link of direct) {
    const key = [link.from, link.to].sort().join(" ");
    pairs.set(key, [...(pairs.get(key) ?? []), link]);
  }
  for (const link of main.links) {
    const a = layout.anchors.get(link.from)!;
    const b = layout.anchors.get(link.to)!;
    const [ra, rb] = [roots[a.root], roots[b.root]];
    const push = (d: string, label?: { x: number; y: number; anchor: "start" | "middle" | "end" }) => {
      layout.links.push({ d, start: link.both, end: true, dashed: link.dashed });
      if (link.label && label) layout.linkLabels.push({ ...label, text: link.label, tier: ROLE.annotation });
    };
    if (link.route === "direct") {
      const inside = siblings(link);
      const parent = parentOf.get(link.from);
      const axis = inside ? childDir.get(parent!.id)! : dir;
      // A nested box's arrow starts and ends at that box, crossing its parent's edge, so field-to-field arrows read as such.
      const top = (id: string) => main.items.some(i => i.id === id);
      const [ea, eb] = inside ? [a.box, b.box] : [top(link.from) ? ra : a.box, top(link.to) ? rb : b.box];
      const order = (id: string) => parent!.children.findIndex(c => c.id === id);
      const forward = inside ? order(link.from) < order(link.to) : a.root < b.root;
      const twins = pairs.get([link.from, link.to].sort().join(" "))!;
      const spread = (twins.indexOf(link) - (twins.length - 1) / 2) * PAIR_SPREAD;
      // Of two arrows side by side, the first is labelled on its outer side, so neither label crosses the other arrow.
      const outside = twins.length > 1 && twins.indexOf(link) === 0;
      if (axis === "vertical") {
        const [lo, hi] = [Math.max(a.box.x, b.box.x), Math.min(a.box.x + a.box.w, b.box.x + b.box.w)];
        const x = (lo < hi ? (lo + hi) / 2 : ea.x + ea.w / 2) + spread;
        const [y1, y2] = forward ? [ea.y + ea.h, eb.y - 1] : [ea.y, eb.y + eb.h + 1];
        push(`M${x},${y1}V${y2}`, outside ? { x: x - 8, y: (y1 + y2) / 2, anchor: "end" } : { x: x + 8, y: (y1 + y2) / 2, anchor: "start" });
      } else {
        const [lo, hi] = [Math.max(a.box.y, b.box.y), Math.min(a.box.y + a.box.h, b.box.y + b.box.h)];
        const y = (lo < hi ? (lo + hi) / 2 : ea.y + ea.h / 2) + spread;
        const [x1, x2] = forward ? [ea.x + ea.w, eb.x - 1] : [ea.x, eb.x + eb.w + 1];
        push(`M${x1},${y}H${x2}`, { x: (x1 + x2) / 2, y: twins.length > 1 && !outside ? y + 11 : y - 9, anchor: "middle" });
      }
      continue;
    }
    const lane = LANE_START + used[link.route]++ * LANE;
    const outer = LANE_START + (main.links.filter(l => l.route === link.route).length - 1) * LANE;
    const midX = (a.x + b.x) / 2;
    if (link.route === "left") push(`M${ra.x},${a.y}H${ra.x - lane}V${b.y}H${rb.x - 1}`, { x: ra.x - outer - 6, y: b.y, anchor: "end" });
    if (link.route === "right") push(`M${ra.x + ra.w},${a.y}H${ra.x + ra.w + lane}V${b.y}H${rb.x + rb.w + 1}`, { x: ra.x + ra.w + outer + 6, y: b.y, anchor: "start" });
    if (link.route === "top") push(`M${a.x},${ra.y}V${ra.y - lane}H${b.x}V${rb.y - 1}`, { x: midX, y: ra.y - outer - 8, anchor: "middle" });
    if (link.route === "bottom") push(`M${a.x},${ra.y + ra.h}V${ra.y + ra.h + lane}H${b.x}V${rb.y + rb.h + 1}`, { x: midX, y: ra.y + ra.h + outer + 11, anchor: "middle" });
  }
  return layout;
};
