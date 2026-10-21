import { type Point, textWidth } from "@/components/frames/FlowPlot/_scene";
import { type BoardItem, type BoardRow, flattenItems } from "@/lib/frames/board";
import { emptyMain, inlineIcons, inlineWidth, MAIN_PAD, type MainLayout, TEXT_INSET } from "./_scene";
import { ROLE, TIER, weightOf } from "./_tokens";

const ROW = 38;
const BAR = 26;
const INDENT = 40;
const ELBOW = 14;
const TREE_MIN = 220;

export const layoutTree = (items: BoardItem[], x0: number, y0: number): MainLayout => {
  const rows = flattenItems(items);
  const inner = Math.max(TREE_MIN, ...rows.map(r => r.depth * INDENT + inlineWidth(r.item.icons.length, ROLE.title) + textWidth(r.item.label, TIER[ROLE.title].text) + 2 * TEXT_INSET + 24));
  const barLeft = x0 + MAIN_PAD;
  const barRight = barLeft + inner;
  const main = emptyMain(inner + 2 * MAIN_PAD, rows.length * ROW - (ROW - BAR) + 2 * MAIN_PAD);
  const at = new Map<BoardRow, Point>(rows.map((r, i) => [r, { x: barLeft + r.depth * INDENT, y: y0 + MAIN_PAD + i * ROW }]));

  for (const r of rows) {
    const { x, y } = at.get(r)!;
    const mid = y + BAR / 2;
    const bar = { x, y, w: barRight - x, h: BAR };
    main.anchors.set(r.item.id, { x: x + bar.w / 2, y: mid, left: x0, right: barRight + 2, gutter: mid + ROW / 2, root: 0, box: bar });
    main.icons.push(...inlineIcons(r.item.icons, x + TEXT_INSET, mid, ROLE.title));
    main.bars.push({
      id: r.item.id,
      x,
      y,
      w: barRight - x,
      h: BAR,
      color: r.item.color ?? "muted",
      text: { x: barRight - TEXT_INSET, y: mid, text: r.item.label, anchor: "end", tier: ROLE.title, weight: weightOf("title") },
    });
    if (r.parent) {
      const parent = at.get(r.parent)!;
      main.elbows.push(`M${parent.x + ELBOW},${parent.y + BAR}V${mid}H${x - 1}`);
    }
  }
  return main;
};
