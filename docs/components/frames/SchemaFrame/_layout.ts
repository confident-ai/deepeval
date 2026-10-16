import {
  blockHeight,
  type FlowScene,
  GROUP_HEAD,
  GROUP_PAD,
  groupHeaderWidth,
  iconRoom,
  LINE,
  normalize,
  PAD_X,
  PAD_Y,
  type SceneEdge,
  type SceneGroup,
  type SceneNode,
  sCurve,
  SMALL,
  smoothPath,
  textWidth,
  wrapWords,
} from "@/components/frames/FlowPlot/_scene";
import { fieldRefOf, type SchemaCardData, type SchemaField, type SchemaObject, type SchemaType } from "@/lib/frames/schema-card";

// SchemaFrame draws objects as containers of field chips, in FlowFrame's
// visual language: chips are nodes, objects are groups, links are edges.

export type Variant = "wide" | "narrow";

const CHIP_GAP = 8;
const SPACING = {
  wide: { wrap: 190, highlight: 64, columns: 40, link: 96 },
  narrow: { wrap: 150, highlight: 36, columns: 24, link: 44 },
};

interface Chip {
  field: SchemaField;
  lines: string[];
  sub: string[];
  textW: number;
  h: number;
}

/** A field as a chip: its name (with `?` when optional) and its example or note under it. */
const chipFor = (field: SchemaField, wrap: number): Chip => {
  const lines = wrapWords(field.optional ? `${field.name}?` : field.name, wrap);
  const sub = [
    ...(field.example ? wrapWords(`“${field.example}”`, wrap, SMALL) : []),
    ...(field.note ? wrapWords(field.note, wrap, SMALL) : []),
  ];
  const textW = Math.max(...lines.map(l => textWidth(l)), ...sub.map(l => textWidth(l, SMALL)));
  return { field, lines, sub, textW, h: blockHeight(lines.length, sub.length) + PAD_Y * 2 };
};

const badgeOf = (object: SchemaObject) =>
  object.subtitle ? { text: object.subtitle, w: textWidth(object.subtitle, SMALL) + 14, color: object.color } : undefined;

/** Narrowest container that fits its chips and its header. */
const columnWidth = (objects: SchemaObject[], chips: Chip[]) => {
  const header = Math.max(...objects.map(o => groupHeaderWidth(o.title, badgeOf(o)?.w)));
  return Math.max(150, ...chips.map(c => c.textW + PAD_X * 2 + GROUP_PAD * 2), header);
};

/**
 * One object's container at (x, y), with a chip (or an empty slot, for null) per
 * row. Rows share heights across columns so fields line up side by side.
 */
const column = (object: SchemaObject, rows: (Chip | null)[], heights: number[], x: number, y: number, w: number) => {
  const nodes: SceneNode[] = [];
  let top = y + GROUP_HEAD + 6;
  rows.forEach((chip, i) => {
    const box = { x: x + GROUP_PAD, y: top, w: w - GROUP_PAD * 2, h: heights[i] };
    nodes.push(
      chip
        ? { ...box, id: fieldRefOf(object, chip.field), lines: chip.lines, sub: chip.sub, color: object.color, shape: "box", dashed: chip.field.optional }
        : { ...box, id: `${object.id}.empty-${i}`, lines: [], color: "muted", shape: "box", ghost: true }
    );
    top += heights[i] + CHIP_GAP;
  });
  const group: SceneGroup = { id: object.id, label: object.title, badge: badgeOf(object), x, y, w, h: top - CHIP_GAP + GROUP_PAD - y };
  return { nodes, group };
};

/** One object, with each highlight as a box joined to the fields it reads. */
const card = (data: SchemaCardData, variant: Variant): FlowScene => {
  const s = SPACING[variant];
  const object = data.objects[0];
  const chips = object.fields.map(f => chipFor(f, s.wrap));
  const w = columnWidth([object], chips);
  const { nodes, group } = column(object, chips, chips.map(c => c.h), 0, 0, w);
  const byRef = new Map(nodes.map(n => [n.id, n]));

  // Wide drawings alternate highlights left and right; narrow ones keep them on the left.
  const boxes = data.highlights.map((h, i) => {
    const side = variant === "wide" && i % 2 === 1 ? "right" : "left";
    const lines = wrapWords(h.label, variant === "wide" ? 140 : 110);
    const width = Math.max(...lines.map(l => textWidth(l))) + iconRoom("gauge") + PAD_X * 2;
    const height = lines.length * LINE + PAD_Y * 2;
    const targets = h.fields.map(ref => byRef.get(ref)!);
    const center = targets.reduce((sum, t) => sum + t.y + t.h / 2, 0) / targets.length;
    return { h, side, lines, width, height, targets, center, y: 0 };
  });
  for (const side of ["left", "right"]) {
    let floor = -Infinity;
    for (const b of boxes.filter(x => x.side === side).sort((p, q) => p.center - q.center)) {
      b.y = Math.max(b.center - b.height / 2, floor);
      floor = b.y + b.height + 12;
    }
  }

  const highlightNodes: SceneNode[] = boxes.map((b, i) => ({
    id: `highlight-${i}`,
    x: b.side === "left" ? group.x - s.highlight - b.width : group.x + group.w + s.highlight,
    y: b.y,
    w: b.width,
    h: b.height,
    lines: b.lines,
    color: b.h.color,
    shape: "box",
    icon: "gauge",
  }));

  // Lines from each highlight to its fields; where two arrive at one chip side, they spread apart.
  const arrivals = new Map<string, number>();
  const counts = new Map<string, number>();
  for (const b of boxes) for (const t of b.targets) counts.set(`${t.id}:${b.side}`, (counts.get(`${t.id}:${b.side}`) ?? 0) + 1);
  const edges: SceneEdge[] = boxes.flatMap((b, i) => {
    const box = highlightNodes[i];
    const start = { x: b.side === "left" ? box.x + box.w : box.x, y: box.y + box.h / 2 };
    return b.targets.map(t => {
      const key = `${t.id}:${b.side}`;
      const k = arrivals.get(key) ?? 0;
      arrivals.set(key, k + 1);
      const spread = (k - ((counts.get(key) ?? 1) - 1) / 2) * 8;
      const end = { x: b.side === "left" ? t.x : t.x + t.w, y: t.y + t.h / 2 + spread };
      return { d: sCurve(start, end), dashed: false, both: false, color: b.h.color };
    });
  });

  return normalize({ width: 0, height: 0, nodes: [...nodes, ...highlightNodes], groups: [group], edges, lifelines: [] });
};

/** Objects side by side with one row per field name, so a field one lacks shows as an empty slot. */
const compare = (data: SchemaCardData, variant: Variant): FlowScene => {
  const s = SPACING[variant];
  const wrap = variant === "narrow" ? s.wrap : data.objects.length > 2 ? 140 : s.wrap;
  const chipsOf = data.objects.map(o => o.fields.map(f => chipFor(f, wrap)));
  const w = columnWidth(data.objects, chipsOf.flat());
  const nodes: SceneNode[] = [];
  const groups: SceneGroup[] = [];

  if (variant === "narrow") {
    // Stacked, so each object lists only its own fields.
    let y = 0;
    data.objects.forEach((o, i) => {
      const col = column(o, chipsOf[i], chipsOf[i].map(c => c.h), 0, y, w);
      nodes.push(...col.nodes);
      groups.push(col.group);
      y += col.group.h + 16;
    });
  } else {
    const names = [...new Set(data.objects.flatMap(o => o.fields.map(f => f.name)))];
    const rows = chipsOf.map(chips => names.map(name => chips.find(c => c.field.name === name) ?? null));
    const heights = names.map((_, r) => Math.max(...rows.map(row => row[r]?.h ?? 0)));
    data.objects.forEach((o, i) => {
      const col = column(o, rows[i], heights, i * (w + s.columns), 0, w);
      nodes.push(...col.nodes);
      groups.push(col.group);
    });
  }
  return normalize({ width: 0, height: 0, nodes, groups, edges: [], lifelines: [] });
};

/** Two objects, rows aligned by link, with an arrow per link. */
const match = (data: SchemaCardData, variant: Variant): FlowScene => {
  const s = SPACING[variant];
  const wrap = variant === "wide" ? 170 : 100;
  const [left, right] = data.objects;
  const chip = (object: SchemaObject, ref: string) => {
    const field = object.fields.find(f => fieldRefOf(object, f) === ref);
    return field ? chipFor(field, wrap) : null;
  };
  const linkedLeft = new Set(data.links.map(l => l.from));
  const linkedRight = new Set(data.links.map(l => l.to));
  const rows: [Chip | null, Chip | null][] = [
    ...data.links.map(l => [chip(left, l.from), chip(right, l.to)] as [Chip | null, Chip | null]),
    ...left.fields.filter(f => !linkedLeft.has(fieldRefOf(left, f))).map(f => [chipFor(f, wrap), null] as [Chip, null]),
    ...right.fields.filter(f => !linkedRight.has(fieldRefOf(right, f))).map(f => [null, chipFor(f, wrap)] as [null, Chip]),
  ];
  const heights = rows.map(([a, b]) => Math.max(a?.h ?? 0, b?.h ?? 0));
  const w = columnWidth([left, right], rows.flat().filter((c): c is Chip => c !== null));
  const l = column(left, rows.map(r => r[0]), heights, 0, 0, w);
  const r = column(right, rows.map(row => row[1]), heights, w + s.link, 0, w);
  const edges: SceneEdge[] = rows.flatMap(([a, b], i) => {
    if (!a || !b) return [];
    const from = l.nodes[i];
    const to = r.nodes[i];
    const y = from.y + from.h / 2;
    return [{ d: smoothPath([{ x: from.x + from.w, y }, { x: to.x, y }]), dashed: false, both: false }];
  });
  return normalize({ width: 0, height: 0, nodes: [...l.nodes, ...r.nodes], groups: [l.group, r.group], edges, lifelines: [] });
};

const LAYOUTS: Record<SchemaType, (data: SchemaCardData, variant: Variant) => FlowScene> = { card, compare, match };

export const layoutSchema = (type: SchemaType, data: SchemaCardData, variant: Variant): FlowScene => LAYOUTS[type](data, variant);
