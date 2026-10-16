import type { ChartType, PartsData } from "@/lib/charts/schema";
import { type Canvas, emptyScene, pointList, type Scene } from "./_scene";

export type PartsType = Extract<ChartType, "pie" | "donut" | "stacked-100" | "waffle" | "treemap" | "funnel">;

const CHAR = 7.4;

const MIN_INSIDE = 0.06;

const formatter = (data: PartsData) => (v: number) =>
  `${v.toLocaleString("en-US", { minimumFractionDigits: data.value.decimals, maximumFractionDigits: data.value.decimals })}${data.value.unit}`;
const percent = (share: number) => (share > 0 && share < 0.01 ? "<1%" : `${Math.round(share * 100)}%`);

export const layoutParts = (type: PartsType, data: PartsData, canvas: Canvas): Scene => {
  switch (type) {
    case "pie":
    case "donut":
      return layoutRing(type, data, canvas);
    case "stacked-100":
      return layoutBar(data, canvas);
    case "waffle":
      return layoutWaffle(data, canvas);
    case "treemap":
      return layoutTreemap(data, canvas);
    case "funnel":
      return layoutFunnel(data, canvas);
  }
};

const shares = (data: PartsData) => {
  const total = data.series.reduce((sum, s) => sum + s.value, 0);
  return { total, share: data.series.map(s => s.value / total) };
};

const layoutRing = (type: PartsType, data: PartsData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const { total, share } = shares(data);
  const cx = width / 2;
  const cy = height / 2;

  const anyOutside = share.some(v => v > 0 && v < MIN_INSIDE);
  const r = anyOutside ? Math.min(height / 2 - 28, width / 2 - 56) : Math.min(height / 2 - 14, width / 2 - 14);
  const inner = type === "donut" ? r * 0.58 : 0;
  const at = (radius: number, a: number) => [cx + radius * Math.sin(a), cy - radius * Math.cos(a)] as const;
  const outside: { mid: number; text: string; i: number }[] = [];

  let start = 0;
  data.series.forEach((ser, i) => {
    const sweep = share[i] * 2 * Math.PI;
    if (sweep <= 0) return;

    const steps = Math.max(2, Math.ceil(sweep / 0.04));
    const outer = Array.from({ length: steps + 1 }, (_, k) => at(r, start + (sweep * k) / steps));
    const back = inner
      ? Array.from({ length: steps + 1 }, (_, k) => at(inner, start + sweep - (sweep * k) / steps))
      : [[cx, cy] as const];
    s.polygons.push({ points: pointList([...outer, ...back]), color: ser.color, i, outline: false, slice: true });
    const mid = start + sweep / 2;
    if (share[i] >= MIN_INSIDE) {
      const [lx, ly] = at(inner ? (r + inner) / 2 : r * 0.64, mid);
      s.labels.push({ x: lx, y: ly + 4.5, text: percent(share[i]), color: "ink", anchor: "middle", i, plain: true });
    } else {
      outside.push({ mid, text: percent(share[i]), i });
    }
    start += sweep;
  });

  const placed = outside.map(o => {
    const [x0, y0] = at(r + 2, o.mid);
    const [x1, y1] = at(r + 11, o.mid);
    const sin = Math.sin(o.mid);
    const side: "top" | "bottom" | "left" | "right" =
      Math.abs(sin) < 0.4 ? (Math.cos(o.mid) > 0 ? "top" : "bottom") : sin > 0 ? "right" : "left";
    return { ...o, x0, y0, x1, y1, side, lx: x1, ly: y1 };
  });
  for (const side of ["left", "right"] as const) {
    const group = placed.filter(p => p.side === side).sort((a, b) => a.ly - b.ly);
    for (let k = 1; k < group.length; k++) group[k].ly = Math.max(group[k].ly, group[k - 1].ly + 15);
  }
  for (const side of ["top", "bottom"] as const) {
    const group = placed.filter(p => p.side === side).sort((a, b) => a.lx - b.lx);
    for (let k = 1; k < group.length; k++) group[k].lx = Math.max(group[k].lx, group[k - 1].lx + 36);
  }
  for (const p of placed) {
    s.lines.push({ d: `M${p.x0} ${p.y0}L${p.x1} ${p.y1}`, color: "muted", i: p.i, width: 1 });
    const label =
      p.side === "top"
        ? { x: p.lx, y: p.ly - 5, anchor: "middle" as const }
        : p.side === "bottom"
          ? { x: p.lx, y: p.ly + 14, anchor: "middle" as const }
          : p.side === "right"
            ? { x: p.lx + 4, y: p.ly + 4.5, anchor: "start" as const }
            : { x: p.lx - 4, y: p.ly + 4.5, anchor: "end" as const };
    s.labels.push({ ...label, text: p.text, color: "ink", i: p.i });
  }
  if (type === "donut") {
    s.labels.push({ x: cx, y: cy + 4, text: formatter(data)(total), color: "ink", anchor: "middle", i: data.series.length, plain: true, large: true });
    s.labels.push({ x: cx, y: cy + 24, text: "total", color: "muted", anchor: "middle", i: data.series.length, plain: true });
  }
  return s;
};

const layoutBar = (data: PartsData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const { share } = shares(data);
  const left = 16;
  const span = width - 32;
  const h = Math.min(64, height * 0.3);
  const y = (height - h) / 2;
  let x = left;

  let lastBelow = { x: -Infinity, row: 0 };
  data.series.forEach((ser, i) => {
    const w = span * share[i];
    s.barGroups.push({ i, from: "left", bars: [{ x, y, w, h, color: ser.color, solid: true }] });
    const cx = x + w / 2;
    const text = percent(share[i]);
    if (w >= text.length * 8 + 12) {
      s.labels.push({ x: cx, y: y + h / 2 + 4.5, text, color: "ink", anchor: "middle", i, plain: true });
    } else if (w > 0) {
      const row = cx - lastBelow.x < 36 ? lastBelow.row + 1 : 0;
      const ly = y + h + 18 + row * 15;
      s.lines.push({ d: `M${cx} ${y + h + 2}V${ly - 11}`, color: "muted", i, width: 1 });
      s.labels.push({ x: cx, y: ly, text, color: "ink", anchor: "middle", i });
      lastBelow = { x: cx, row };
    }
    x += w;
  });
  return s;
};

const squares = (share: number[]) => {
  const raw = share.map(v => v * 100);
  const counts = raw.map(Math.floor);
  const order = raw.map((v, i) => [v - Math.floor(v), i] as const).sort((a, b) => b[0] - a[0]);
  for (let k = 0; counts.reduce((a, b) => a + b, 0) < 100; k++) counts[order[k % order.length][1]]++;
  return counts;
};

const layoutWaffle = (data: PartsData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const counts = squares(shares(data).share);
  const cols = width >= 500 ? 20 : 10;
  const rows = 100 / cols;
  const gap = 3;
  const cell = Math.min((width - 32 - gap * (cols - 1)) / cols, (height - 32 - gap * (rows - 1)) / rows);
  const x0 = (width - (cell * cols + gap * (cols - 1))) / 2;
  const y0 = (height - (cell * rows + gap * (rows - 1))) / 2;
  let k = 0;
  data.series.forEach((ser, i) => {
    for (let j = 0; j < counts[i]; j++, k++) {
      const row = Math.floor(k / cols);
      const col = k % cols;
      s.barGroups.push({
        i: row,
        from: "bottom",
        bars: [{ x: x0 + col * (cell + gap), y: y0 + row * (cell + gap), w: cell, h: cell, color: ser.color, solid: true }],
      });
    }
  });
  return s;
};

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const partition = (items: { value: number; index: number }[], r: Rect): { index: number; rect: Rect }[] => {
  if (items.length === 1) return [{ index: items[0].index, rect: r }];
  const total = items.reduce((sum, it) => sum + it.value, 0);
  let acc = 0;
  let cut = 1;
  for (; cut < items.length - 1; cut++) {
    if (acc + items[cut - 1].value >= total / 2) break;
    acc += items[cut - 1].value;
  }
  acc = items.slice(0, cut).reduce((sum, it) => sum + it.value, 0);
  const f = total ? acc / total : 0.5;
  const [a, b] =
    r.w >= r.h
      ? [{ ...r, w: r.w * f }, { ...r, x: r.x + r.w * f, w: r.w * (1 - f) }]
      : [{ ...r, h: r.h * f }, { ...r, y: r.y + r.h * f, h: r.h * (1 - f) }];
  return [...partition(items.slice(0, cut), a), ...partition(items.slice(cut), b)];
};

const layoutTreemap = (data: PartsData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const { share } = shares(data);
  const items = data.series
    .map((ser, index) => ({ value: ser.value, index }))
    .filter(it => it.value > 0)
    .sort((a, b) => b.value - a.value);
  const fmt = formatter(data);
  for (const { index: i, rect } of partition(items, { x: 8, y: 8, w: width - 16, h: height - 16 })) {
    const ser = data.series[i];
    s.barGroups.push({ i, from: "bottom", bars: [{ ...rect, color: ser.color, solid: true }] });
    if (rect.w >= Math.max(64, ser.label.length * CHAR + 16) && rect.h >= 44) {
      s.labels.push({ x: rect.x + 10, y: rect.y + 22, text: ser.label, color: "ink", anchor: "start", i, plain: true });
      s.labels.push({ x: rect.x + 10, y: rect.y + 40, text: `${fmt(ser.value)} · ${percent(share[i])}`, color: "ink", anchor: "start", i, plain: true });
    } else if (rect.w >= 40 && rect.h >= 24) {
      s.labels.push({ x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 + 4.5, text: percent(share[i]), color: "ink", anchor: "middle", i, plain: true });
    }
  }
  return s;
};

const layoutFunnel = (data: PartsData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const fmt = formatter(data);
  const first = data.series[0].value || 1;
  const labelW = Math.min(width * 0.32, Math.max(...data.series.map(ser => ser.label.length)) * CHAR);
  const left = labelW + 18;
  const right = width - 112;
  const rowH = (height - 16) / data.series.length;
  const center = (left + right) / 2;
  data.series.forEach((ser, i) => {
    const w = Math.max(2, ((right - left) * ser.value) / first);
    const y = 8 + rowH * i;
    s.barGroups.push({ i, from: "bottom", bars: [{ x: center - w / 2, y: y + 3, w, h: rowH - 6, color: ser.color, solid: true }] });
    s.categoryLabels.push({ x: labelW + 6, y: y + rowH / 2 + 4.5, lines: [ser.label], anchor: "end" });
    s.labels.push({ x: right + 10, y: y + rowH / 2 + 4.5, text: `${fmt(ser.value)} · ${percent(ser.value / first)}`, color: ser.color, anchor: "start", i });
  });
  return s;
};
