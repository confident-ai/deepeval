import { z } from "zod";
import { type ChartColor, chartColor } from "@/lib/charts/schema";
import { ICON_REF, iconProblem } from "./board-icons";

const nonEmpty = z.string().trim().min(1);
const itemId = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "use lowercase-kebab-case item ids");

export const ICON_ARRANGEMENTS = ["grid", "line", "zigzag-h", "zigzag-v", "circle", "scatter"] as const;
export type IconArrangement = (typeof ICON_ARRANGEMENTS)[number];
export const DIRECTIONS = ["vertical", "horizontal"] as const;
export type Direction = (typeof DIRECTIONS)[number];
export const ITEM_SPLITS = ["cells", "boxes"] as const;
export const ITEM_SHAPES = ["box", "circle"] as const;
export const ITEM_BORDERS = ["solid", "dashed", "none"] as const;
export const BOARD_SIZES = ["xs", "sm", "md", "lg"] as const;
export type BoardSize = (typeof BOARD_SIZES)[number];
export const LABEL_PLACES = ["head", "above", "below"] as const;
export type LabelPlace = (typeof LABEL_PLACES)[number];
export const ICON_PLACES = ["body", "label"] as const;

const iconRef = z.string().regex(ICON_REF, 'write icons as "brand:<slug>", "icon:<name>" or "media:<id>", optionally with "@<color>"');

const iconFields = {
  icons: z.array(iconRef).default([]),
  arrange: z.enum(ICON_ARRANGEMENTS).optional(),
};

export interface BoardLine {
  text: string;
  icons: string[];
}

export interface BoardItem {
  id: string;
  label: string;
  color?: ChartColor;
  icons: string[];
  arrange?: IconArrangement;
  size?: BoardSize;
  labelAt?: LabelPlace;
  iconsAt?: (typeof ICON_PLACES)[number];
  lines: BoardLine[];
  shape?: (typeof ITEM_SHAPES)[number];
  border?: (typeof ITEM_BORDERS)[number];
  direction?: Direction;
  split?: (typeof ITEM_SPLITS)[number];
  columns?: number;
  children: BoardItem[];
}

const textLine = z.strictObject({ text: nonEmpty, icons: z.array(iconRef).default([]) });

const boardItem: z.ZodType<BoardItem, unknown> = z.lazy(() =>
  z.strictObject({
    id: itemId,
    label: nonEmpty,
    color: chartColor.optional(),
    ...iconFields,
    size: z.enum(BOARD_SIZES).optional(),
    labelAt: z.enum(LABEL_PLACES).optional(),
    iconsAt: z.enum(ICON_PLACES).optional(),
    lines: z.array(textLine).default([]),
    shape: z.enum(ITEM_SHAPES).optional(),
    border: z.enum(ITEM_BORDERS).optional(),
    direction: z.enum(DIRECTIONS).optional(),
    split: z.enum(ITEM_SPLITS).optional(),
    columns: z.number().int().min(2).optional(),
    children: z.array(boardItem).default([]),
  })
);

export const NOTE_ARROWS = ["out", "in", "both", "none"] as const;
export const NOTE_SHAPES = ["box", "text"] as const;

const noteFields = {
  for: itemId.optional(),
  label: nonEmpty,
  color: chartColor.optional(),
  shape: z.enum(NOTE_SHAPES).default("box"),
  arrow: z.enum(NOTE_ARROWS).default("out"),
  arrowLabel: nonEmpty.optional(),
  size: z.enum(BOARD_SIZES).optional(),
  ...iconFields,
};

const noteColumn = z.strictObject({
  heading: nonEmpty.optional(),
  items: z.array(z.strictObject(noteFields)).min(1),
});

const bottomNotes = z.strictObject({
  items: z.array(z.strictObject(noteFields)).min(1),
});

export const LINK_ROUTES = ["direct", "left", "right", "top", "bottom"] as const;
export type LinkRoute = (typeof LINK_ROUTES)[number];

const link = z.strictObject({
  from: itemId,
  to: itemId,
  route: z.enum(LINK_ROUTES).default("direct"),
  label: nonEmpty.optional(),
  both: z.boolean().default(false),
  dashed: z.boolean().default(false),
});

export const LINE_ENDS = ["magnifier", "circle", "dot", "arrow", "none"] as const;
export type LineEnd = (typeof LINE_ENDS)[number];
export const LINE_STYLES = ["solid", "dashed"] as const;
export const PULLOUT_ROUTES = ["gap", "straight"] as const;
export const PULLOUT_PLACES = ["above", "level", "below"] as const;
export const PULLOUT_SHAPES = ["circle", "none"] as const;

const pulloutLine = z.strictObject({
  style: z.enum(LINE_STYLES).default("solid"),
  color: chartColor.optional(),
  start: z.enum(LINE_ENDS).default("magnifier"),
  end: z.enum(LINE_ENDS).default("none"),
  label: nonEmpty.optional(),
  icons: z.array(iconRef).default([]),
});

const pullout = z.strictObject({
  for: itemId.optional(),
  shape: z.enum(PULLOUT_SHAPES).default("circle"),
  heading: nonEmpty.optional(),
  route: z.enum(PULLOUT_ROUTES).default("gap"),
  place: z.enum(PULLOUT_PLACES).default("level"),
  size: z.enum(BOARD_SIZES).optional(),
  line: z.union([pulloutLine, z.array(pulloutLine).min(1).max(2)]).prefault({}),
  ...iconFields,
  items: z.array(z.strictObject({ label: nonEmpty, color: chartColor.default("muted"), icons: z.array(iconRef).default([]) })).default([]),
});

/** A venn member: text drawn in the region of the sets it's `in` (one set, or both of two overlapping sets). */
const member = z.strictObject({
  in: z.array(itemId).min(1).max(2),
  label: nonEmpty.optional(),
  color: chartColor.optional(),
  lines: z.array(textLine).min(1),
});

export const BOARD_SIDES = ["left", "right"] as const;
export type BoardSide = (typeof BOARD_SIDES)[number];

export const BOARD_RADII = ["none", "small", "medium", "large"] as const;
export type BoardRadius = (typeof BOARD_RADII)[number];

export const BOARD_FILLS = ["hatch", "tint"] as const;

export const boardData = z.strictObject({
  radius: z.enum(BOARD_RADII).default("medium"),
  /** The border around the whole board; with "none" its space and fill stay. */
  panel: z.enum(["solid", "none"]).default("solid"),
  fill: z.enum(BOARD_FILLS).default("hatch"),
  heading: z
    .strictObject({
      text: nonEmpty,
      align: z.enum(["left", "center", "right"]).default("left"),
    })
    .optional(),
  main: z.strictObject({
    border: z.enum(["solid", "dashed", "none"]).default("dashed"),
    direction: z.enum(DIRECTIONS).optional(),
    columns: z.number().int().min(2).optional(),
    items: z.array(boardItem).min(1),
    links: z.array(link).default([]),
    members: z.array(member).default([]),
  }),
  notes: z
    .strictObject({ left: noteColumn.optional(), right: noteColumn.optional(), bottom: bottomNotes.optional() })
    .default({}),
  pullouts: z
    .strictObject({ left: z.array(pullout).min(1).optional(), right: z.array(pullout).min(1).optional() })
    .default({}),
});

export type BoardData = z.infer<typeof boardData>;
export type BoardNoteColumn = z.infer<typeof noteColumn>;
export type BoardNote = BoardNoteColumn["items"][number];
export type BoardLink = z.infer<typeof link>;
export type BoardMember = z.infer<typeof member>;
export const NOTE_SIDES = ["left", "right", "bottom"] as const;
export type BoardPullout = z.infer<typeof pullout>;
export type BoardPulloutLine = z.infer<typeof pulloutLine>;

export const pulloutLines = (pullout: BoardPullout): BoardPulloutLine[] => (Array.isArray(pullout.line) ? pullout.line : [pullout.line]);

export const BOARD_TYPES = ["tree", "groups", "venn"] as const;
export type BoardType = (typeof BOARD_TYPES)[number];

export interface BoardRow {
  item: BoardItem;
  depth: number;
  parent?: BoardRow;
}

export const flattenItems = (items: BoardItem[], depth = 0, parent?: BoardRow, rows: BoardRow[] = []): BoardRow[] => {
  for (const item of items) {
    const row: BoardRow = { item, depth, parent };
    rows.push(row);
    flattenItems(item.children, depth + 1, row, rows);
  }
  return rows;
};

const repeated = (values: string[]) => [...new Set(values.filter((v, i) => values.indexOf(v) !== i))];

export const boardProblems = (data: BoardData): string[] => {
  const problems: string[] = [];
  const rows = flattenItems(data.main.items);
  const ids = rows.map(r => r.item.id);
  const known = new Set(ids);
  for (const id of repeated(ids)) problems.push(`item id "${id}" is used more than once`);
  const checkIcons = (icons: string[], at: string) =>
    icons.forEach((ref, i) => {
      const problem = iconProblem(ref);
      if (problem) problems.push(`${at}.icons.${i}": ${problem}`);
    });
  const parentOf = new Map(rows.flatMap(r => r.item.children.map((child, index) => [child.id, { parent: r.item, index }] as const)));
  for (const { item } of rows) {
    checkIcons(item.icons, `item "${item.id}" "`);
    item.lines.forEach((line, i) => checkIcons(line.icons, `item "${item.id}" "lines.${i}`));
    if (item.shape === "circle" && item.children.length) problems.push(`item "${item.id}": a circle can't hold children`);
    if (item.split && !item.children.length) problems.push(`item "${item.id}": "split" needs children`);
    if (item.columns && !item.children.length) problems.push(`item "${item.id}": "columns" needs children`);
    if (item.columns && data.main.links.some(l => item.children.some(c => c.id === l.from) && item.children.some(c => c.id === l.to)))
      problems.push(`item "${item.id}": links can't join boxes in a grid; drop "columns" or the links between its children`);
    if (item.labelAt === "below" && item.children.length) problems.push(`item "${item.id}": a label below can't sit under children; use "head" or "above"`);
    if (item.labelAt === "head" && item.shape === "circle") problems.push(`item "${item.id}": a circle has no head; use labelAt "above" or "below"`);
    if (item.iconsAt === "label" && !item.icons.length) problems.push(`item "${item.id}": "iconsAt" needs "icons"`);
    if (item.iconsAt === "label" && item.arrange) problems.push(`item "${item.id}": "arrange" places icons in the body; drop it with "iconsAt": "label"`);
  }
  for (const side of NOTE_SIDES) {
    const notes = data.notes[side]?.items ?? [];
    notes.forEach((note, i) => {
      const at = `"notes.${side}.items.${i}`;
      if (note.for && !known.has(note.for)) problems.push(`${at}.for" names unknown item "${note.for}"`);
      if (note.arrange && note.shape === "box") problems.push(`${at}": a box shows its icons inline; "arrange" needs shape "text"`);
      checkIcons(note.icons, at);
    });
    for (const id of repeated(notes.map(n => n.for ?? "the whole board"))) problems.push(`"notes.${side}" annotates ${id === "the whole board" ? id : `"${id}"`} more than once`);
  }
  for (const side of BOARD_SIDES) {
    const pullouts = data.pullouts[side] ?? [];
    pullouts.forEach((c, i) => {
      const at = `"pullouts.${side}.${i}`;
      if (c.for && !known.has(c.for)) problems.push(`${at}.for" names unknown item "${c.for}"`);
      if (!c.items.length && !c.icons.length && !c.heading) problems.push(`${at}" needs a "heading", "items" or "icons"`);
      checkIcons(c.icons, at);
      c.items.forEach((item, j) => checkIcons(item.icons, `${at}.items.${j}`));
      const lines = pulloutLines(c);
      lines.forEach((line, j) => checkIcons(line.icons, `${at}.line${lines.length > 1 ? `.${j}` : ""}`));
      if (lines.length > 1 && c.route !== "straight") problems.push(`${at}": side-by-side lines need route "straight"`);
    });
    for (const id of repeated(pullouts.map(c => c.for ?? "the whole board"))) problems.push(`"pullouts.${side}" points at ${id === "the whole board" ? id : `"${id}"`} more than once`);
  }
  const root = new Map<string, number>();
  data.main.items.forEach((item, i) => flattenItems([item]).forEach(r => root.set(r.item.id, i)));
  const sides = data.main.direction === "horizontal" ? ["top", "bottom"] : ["left", "right"];
  if (data.main.columns && data.main.links.length) problems.push('"main.columns" lays the boxes out in a grid, which links can\'t join; drop the columns or "main.links"');
  data.main.links.forEach((l, i) => {
    const at = `"main.links.${i}"`;
    for (const end of [l.from, l.to]) if (!known.has(end)) problems.push(`${at} names unknown item "${end}"`);
    if (l.from === l.to) problems.push(`${at} links "${l.from}" to itself`);
    if (!known.has(l.from) || !known.has(l.to)) return;
    const [pa, pb] = [parentOf.get(l.from), parentOf.get(l.to)];
    const siblings = pa && pb && pa.parent === pb.parent && pa.parent.split === "boxes" && Math.abs(pa.index - pb.index) === 1;
    if (l.route === "direct" && !siblings && Math.abs(root.get(l.from)! - root.get(l.to)!) !== 1) {
      problems.push(`${at}: "direct" links neighbouring boxes (top-level, or children of a "split": "boxes" item); go around with route ${sides.map(s => `"${s}"`).join(" or ")}`);
    }
    if (l.route !== "direct" && !sides.includes(l.route)) {
      problems.push(`${at}: route "${l.route}" doesn't fit ${data.main.direction ?? "vertical"} boxes; use "direct", ${sides.map(s => `"${s}"`).join(" or ")}`);
    }
  });
  return problems;
};

const VENN_UNUSED = ["arrange", "labelAt", "iconsAt", "direction", "shape", "split", "columns"] as const;

const vennProblems = (data: BoardData): string[] => {
  const problems: string[] = [];
  const { items, members } = data.main;
  if (data.main.direction || data.main.columns) problems.push('venn doesn\'t use "main.direction" or "main.columns"; it places the sets itself');
  if (data.main.links.length) problems.push('venn has no links; "main.links" are for groups');
  if (items.length > 2) problems.push(`venn draws one set, or two overlapping; it has ${items.length} top-level items (nest sets inside one instead)`);
  if (items.length === 2 && items.some(i => i.children.length)) problems.push("two overlapping sets hold members, not sets; to nest sets, use one top-level set");
  const ids = new Set(flattenItems(items).map(r => r.item.id));
  for (const { item } of flattenItems(items)) {
    const unused: string[] = VENN_UNUSED.filter(key => item[key]);
    if (item.icons.length) unused.push("icons");
    if (item.lines.length) unused.push("lines");
    if (unused.length) problems.push(`venn sets don't use ${unused.map(k => `"${k}"`).join(" or ")} (item "${item.id}"); put text in "main.members"`);
  }
  members.forEach((m, i) => {
    for (const id of m.in) if (!ids.has(id)) problems.push(`"main.members.${i}.in" names unknown set "${id}"`);
    if (m.in.length === 2 && !(items.length === 2 && m.in.every(id => items.some(i => i.id === id))))
      problems.push(`"main.members.${i}" is in two sets; that needs the two top-level, overlapping sets`);
  });
  return problems;
};

export const boardTypeProblems = (type: BoardType, data: BoardData): string[] => {
  if (type !== "venn" && data.main.members.length) return [`"main.members" are for venn boards, not ${type}`];
  if (type === "venn") return vennProblems(data);
  if (type === "groups") {
    const rows = flattenItems(data.main.items);
    return rows.flatMap(({ item, parent }) =>
      parent && parent.item.split !== "boxes" && (item.shape || item.border || item.color)
        ? [`item "${item.id}" is a cell, so it has no box to shape, border or tint; set "split": "boxes" on its parent`]
        : []
    );
  }
  if (type !== "tree") return [`unknown type "${type}"`];
  const problems: string[] = [];
  if (data.main.direction) problems.push('tree doesn\'t use "main.direction"; it\'s for groups');
  if (data.main.columns) problems.push('tree doesn\'t use "main.columns"; it\'s for groups');
  if (data.main.links.length) problems.push('tree draws its own arrows; "main.links" are for groups');
  for (const { item } of flattenItems(data.main.items)) {
    const unused: string[] = (["arrange", "size", "labelAt", "iconsAt", "direction", "shape", "border", "split", "columns"] as const).filter(key => item[key]);
    if (item.lines.length) unused.push("lines");
    if (unused.length) problems.push(`tree rows don't use ${unused.map(k => `"${k}"`).join(" or ")} (item "${item.id}"); those are for groups`);
  }
  return problems;
};
