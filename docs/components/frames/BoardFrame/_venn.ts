import { textWidth, wrapWords } from "@/components/frames/FlowPlot/_scene";
import type { BoardData, BoardItem, BoardMember, BoardSize } from "@/lib/frames/board";
import { emptyMain, MAIN_PAD, type MainLayout } from "./_scene";
import { ROLE, TIER, weightOf } from "./_tokens";

/** Room between a circle's edge and what's inside it. */
const MARGIN = 18;
/** Between circles side by side, and between members stacked in a region. */
const GAP = 16;
const MIN_SET = 72;
const MIN_LENS = 64;
const TITLE_ROOM = 10;
/** Set titles wrap at this width, so a long name doesn't inflate its circle: nested sets narrower than top-level ones. */
const TITLE_WRAP = { nested: 96, top: 200 };
/** A member is an ellipse around its text: padding, then how much wider and taller the ellipse is than that box. */
const MEMBER_PAD = { x: 18, y: 10 };
const MEMBER_ROUND = { x: 1.3, y: 1.45 };

const BODY = TIER[ROLE.body];

interface Member {
  member: BoardMember;
  w: number;
  h: number;
}

interface Stack {
  members: Member[];
  w: number;
  h: number;
}

interface SetSize {
  item: BoardItem;
  depth: number;
  tier: BoardSize;
  titleLines: string[];
  band: number;
  kids: SetSize[];
  own: Stack;
  row: { w: number; h: number };
  /** Child sets without members of their own sit evenly on a ring this far from the middle. */
  ring: number;
  d: number;
}

const memberOf = (member: BoardMember): Member => {
  const w = Math.max(...member.lines.map(l => textWidth(l.text, BODY.text))) + 2 * MEMBER_PAD.x;
  const h = member.lines.length * BODY.line + 2 * MEMBER_PAD.y;
  return { member, w: w * MEMBER_ROUND.x, h: h * MEMBER_ROUND.y };
};

const stackOf = (members: BoardMember[]): Stack => {
  const sized = members.map(memberOf);
  return {
    members: sized,
    w: Math.max(0, ...sized.map(m => m.w)),
    h: sized.reduce((sum, m) => sum + m.h, 0) + Math.max(0, sized.length - 1) * GAP,
  };
};

/** Sets, members and how they relate: a venn's sets are the board's items; nested items are sets inside their parent. */
export const layoutVenn = (main: BoardData["main"], x0: number, y0: number): MainLayout => {
  const members = main.members;
  const only = (id: string) => members.filter(m => m.in.length === 1 && m.in[0] === id);

  const measure = (item: BoardItem, depth: number): SetSize => {
    const tier = item.size ?? (depth ? ROLE.label : ROLE.title);
    // Wrap between words only: never narrower than the longest word, so no word is split.
    const longestWord = Math.max(...item.label.split(/\s+/).map(word => textWidth(word, TIER[tier].text)));
    const titleLines = wrapWords(item.label, Math.max(depth ? TITLE_WRAP.nested : TITLE_WRAP.top, longestWord), TIER[tier].text);
    const band = titleLines.length * TIER[tier].line + TITLE_ROOM;
    const kids = item.children.map(child => measure(child, depth + 1));
    const own = stackOf(only(item.id));
    const row = { w: kids.reduce((sum, k) => sum + k.d, 0) + Math.max(0, kids.length - 1) * GAP, h: Math.max(0, ...kids.map(k => k.d)) };
    const label = Math.max(...titleLines.map(line => textWidth(line, TIER[tier].text)));
    const base = { item, depth, tier, titleLines, band, kids, own, row, ring: 0 };
    if (!kids.length && !own.members.length) return { ...base, d: Math.max(MIN_SET, Math.hypot(label, band) + MARGIN) };
    if (!own.members.length) {
      // Child sets on a ring (one in the middle, two side by side, three in a triangle…), the title above them.
      const kd = row.h;
      // Sibling sets on a ring are drawn the same size, so the ring reads as a set of equals.
      for (const kid of kids) kid.d = kd;
      const ring = kids.length === 1 ? 0 : (kd + GAP) / (2 * Math.sin(Math.PI / kids.length));
      let r = ring + kd / 2 + MARGIN + band / 2;
      // The title sits near the top, where the circle is narrowest: grow it until the title fits that chord.
      while (2 * Math.sqrt(Math.max(0, r * r - (r - MARGIN - band / 2) ** 2)) < label + MARGIN && r < 2000) r += 2;
      return { ...base, ring, d: 2 * r };
    }
    const w = Math.max(label, row.w, own.w);
    const h = band + row.h + (row.h && own.h ? GAP : 0) + own.h;
    return { ...base, d: Math.hypot(w, h) + 2 * MARGIN };
  };

  const placeStack = (layout: MainLayout, stack: Stack, cx: number, top: number) => {
    let y = top;
    for (const { member, w, h } of stack.members) {
      layout.boxes.push({ id: `member-${layout.boxes.length}`, x: cx - w / 2, y, w, h, color: member.color, shape: "ellipse", depth: 1 });
      const textTop = y + (h - member.lines.length * BODY.line) / 2;
      member.lines.forEach((line, i) =>
        layout.labels.push({ x: cx, y: textTop + i * BODY.line + BODY.line / 2, text: line.text, anchor: "middle", tier: ROLE.body, role: "body" })
      );
      y += h + GAP;
    }
  };

  /** A set's title lines, centered on (x, y). */
  const title = (layout: MainLayout, s: SetSize, x: number, y: number) => {
    const { line } = TIER[s.tier];
    const role = s.depth ? "label" : "title";
    s.titleLines.forEach((text, i) =>
      layout.labels.push({ x, y: y + (i - (s.titleLines.length - 1) / 2) * line, text, anchor: "middle", tier: s.tier, role, weight: weightOf(role) })
    );
  };

  const anchorSet = (layout: MainLayout, s: SetSize, cx: number, cy: number, r: number) =>
    layout.anchors.set(s.item.id, { x: cx, y: cy, left: cx - r - 2, right: cx + r + 2, gutter: cy + r + MAIN_PAD / 2, root: 0, box: { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r } });

  /** A set and everything inside it, centered on (cx, cy). */
  const placeSet = (layout: MainLayout, s: SetSize, cx: number, cy: number) => {
    const { color, border } = s.item;
    layout.boxes.push({ id: s.item.id, x: cx - s.d / 2, y: cy - s.d / 2, w: s.d, h: s.d, color, border, shape: "circle", depth: s.depth });
    anchorSet(layout, s, cx, cy, s.d / 2);
    if (!s.kids.length && !s.own.members.length) {
      title(layout, s, cx, cy);
      return;
    }
    if (!s.own.members.length) {
      const r = s.d / 2;
      title(layout, s, cx, cy - r + MARGIN + s.band / 2);
      const middle = cy + s.band / 2;
      s.kids.forEach((kid, i) => {
        const angle = -Math.PI / 2 + Math.PI / s.kids.length + (i * 2 * Math.PI) / s.kids.length;
        placeSet(layout, kid, cx + s.ring * Math.cos(angle), middle + s.ring * Math.sin(angle));
      });
      return;
    }
    const h = s.band + s.row.h + (s.row.h && s.own.h ? GAP : 0) + s.own.h;
    const top = cy - h / 2;
    title(layout, s, cx, top + s.band / 2);
    let x = cx - s.row.w / 2;
    for (const kid of s.kids) {
      placeSet(layout, kid, x + kid.d / 2, top + s.band + s.row.h / 2);
      x += kid.d + GAP;
    }
    placeStack(layout, s.own, cx, top + s.band + s.row.h + (s.row.h && s.own.h ? GAP : 0));
  };

  const [a, b] = main.items.map(item => measure(item, 0));
  if (!b) {
    const layout = emptyMain(a.d + 2 * MAIN_PAD, a.d + 2 * MAIN_PAD);
    placeSet(layout, a, x0 + MAIN_PAD + a.d / 2, y0 + MAIN_PAD + a.d / 2);
    return layout;
  }

  // Two overlapping sets: each one's own members in its crescent, shared members in the lens.
  const both = stackOf(members.filter(m => m.in.length === 2));
  const [onlyA, onlyB] = [a.own, b.own];
  let apart = Math.max(onlyA.w, onlyB.w, MIN_SET) + 2 * MARGIN;
  const lens = Math.max(both.w + 2 * MARGIN, MIN_LENS);
  // Grow the circles until every member's left, right, top and bottom sit inside the circles it belongs to.
  const offsets = (stack: Stack) => {
    let y = -stack.h / 2;
    return stack.members.map(m => {
      const dy = y + m.h / 2;
      y += m.h + GAP;
      return { m, dy };
    });
  };
  const RIM = Array.from({ length: 24 }, (_, i) => (i * Math.PI) / 12);
  const inside = (r: number, dx: number, dy: number, m: Member) =>
    RIM.every(a => Math.hypot(dx + (Math.cos(a) * m.w) / 2, dy + (Math.sin(a) * m.h) / 2) <= r - MARGIN);
  const crescentsFit = (r: number) =>
    offsets(onlyA).every(({ m, dy }) => inside(r, -r + apart / 2, dy, m)) && offsets(onlyB).every(({ m, dy }) => inside(r, r - apart / 2, dy, m));
  const lensFits = (r: number) => offsets(both).every(({ m, dy }) => inside(r, apart / 2, dy, m) && inside(r, -apart / 2, dy, m));
  // A crescent that's too tight needs the circles further apart; a lens that's too tight needs bigger circles.
  let r = (apart + lens) / 2;
  for (let step = 0; step < 1000 && !(crescentsFit(r) && lensFits(r)); step++) {
    if (!crescentsFit(r)) {
      apart += 2;
      r = Math.max(r, (apart + lens) / 2);
    } else r += 2;
  }
  const band = Math.max(a.band, b.band);
  const layout = emptyMain(2 * r + apart + 2 * MAIN_PAD, band + 2 * r + 2 * MAIN_PAD);
  const cy = y0 + MAIN_PAD + band + r;
  const [ax, bx] = [x0 + MAIN_PAD + r, x0 + MAIN_PAD + r + apart];
  for (const [s, cx] of [[a, ax], [b, bx]] as const) {
    layout.boxes.push({ id: s.item.id, x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, color: s.item.color, border: s.item.border, shape: "circle", depth: 0, wash: true });
    anchorSet(layout, s, cx, cy, r);
    title(layout, s, cx, cy - r - band / 2);
  }
  placeStack(layout, onlyA, ax - r + apart / 2, cy - onlyA.h / 2);
  placeStack(layout, onlyB, bx + r - apart / 2, cy - onlyB.h / 2);
  placeStack(layout, both, (ax + bx) / 2, cy - both.h / 2);
  return layout;
};
