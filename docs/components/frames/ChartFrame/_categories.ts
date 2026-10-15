import { wrapLabel as wrap, wrapFullLabel, labelWidth } from "./_labels";
import { runningTotals, valueDomain } from "@/lib/charts/domain";
import type { CategoryData, ChartColor, ChartType } from "@/lib/charts/schema";
import { niceTicks } from "./_scale";
import { type Canvas, emptyScene, type Scene } from "./_scene";

export type CategoryType = Extract<
  ChartType,
  "grouped-bars" | "stacked-bars" | "horizontal-bars" | "lollipop" | "dumbbell" | "slope" | "radar" | "waterfall"
>;

export const layoutCategories = (type: CategoryType, data: CategoryData, canvas: Canvas): Scene => {
  switch (type) {
    case "horizontal-bars":
    case "dumbbell":
      return layoutRows(type, data, canvas);
    case "radar":
      return layoutRadar(data, canvas);
    default:
      return layoutColumns(type, data, canvas);
  }
};

const format = (data: CategoryData) => (v: number) => `${v.toFixed(data.y.decimals)}${data.y.unit}`;
const tickText = (data: CategoryData) => (v: number) => `${+v.toFixed(6)}${data.y.unit}`;

const columnPlan = (type: CategoryType, data: CategoryData, requestedWidth: number) => {
  const categories = type === "waterfall" ? [...data.x.categories, "Total"] : data.x.categories;
  const { min, max, step } = valueDomain(type, data);
  const tickWidth = Math.max(...niceTicks(min, max, step).map(v => labelWidth(tickText(data)(v), 11)));
  const baseLeft = type === "slope" ? 16 : Math.min(requestedWidth * 0.3, Math.max(52, tickWidth + 12));
  const right = type === "slope" ? 16 : 10;
  const slot = (requestedWidth - baseLeft - right) / categories.length;
  const wrapped = categories.map(c => wrapFullLabel(c, Math.max(16, slot - 6)));
  // A long unbroken word must also rotate, rather than being split in two.
  const rotate = wrapped.some(lines => lines.length > 2) || categories.some(c => c.split(/\s+/).some(w => labelWidth(w) > slot - 6));
  const projection = Math.max(...categories.map(c => labelWidth(c))) / Math.SQRT2;
  const left = rotate ? Math.max(baseLeft, projection + 12) : baseLeft;
  const width = Math.max(requestedWidth, left + right + categories.length * (rotate ? 32 : 40));
  const lines = rotate ? categories.map(c => [c]) : wrapped;
  const bottom = rotate ? Math.ceil(projection + 32) : 14 + 15 * Math.max(...lines.map(l => l.length));
  return { categories, min, max, step, left, right, width, lines, bottom, rotate };
};

const rowPlan = (type: CategoryType, data: CategoryData, width: number) => {
  const labelArea = Math.min(width * 0.36, Math.max(...data.x.categories.map(c => labelWidth(c))) + 4);
  const lines = data.x.categories.map(c => wrapFullLabel(c, labelArea));
  const marks = type === "horizontal-bars" ? 20 + 18 * data.series.length : 44;
  const rowHeight = Math.max(marks, 15 * Math.max(...lines.map(l => l.length)) + 16);
  return { labelArea, lines, height: Math.max(180, 50 + data.x.categories.length * rowHeight) };
};

/** Reserve label space without taking height away from the data plot. */
export const categoryCanvas = (type: CategoryType, data: CategoryData, width: number): Canvas => {
  if (type === "horizontal-bars" || type === "dumbbell") return { width, height: rowPlan(type, data, width).height };
  const plan = columnPlan(type, data, width);
  return { width: plan.width, height: 276 + plan.bottom, layoutWidth: width, ...(plan.width > width ? { minWidth: plan.width } : {}) };
};

const layoutColumns = (type: CategoryType, data: CategoryData, canvas: Canvas): Scene => {
  const s = emptyScene();
  const fmt = format(data);
  const slope = type === "slope";
  const { categories, min, max, step, left, right, width, lines: labelLines, bottom: bottomPad, rotate } = columnPlan(type, data, canvas.layoutWidth ?? canvas.width);
  const height = Math.max(canvas.height, 276 + bottomPad);
  const colWidth = (width - left - right) / categories.length;
  const plot = { left, right: width - right, top: 30, bottom: height - bottomPad };
  const sy = (v: number) => plot.bottom - ((v - min) / (max - min)) * (plot.bottom - plot.top);
  const centers = slope
    ? [plot.left + (plot.right - plot.left) * 0.3, plot.left + (plot.right - plot.left) * 0.7]
    : categories.map((_, c) => plot.left + colWidth * (c + 0.5));

  if (!slope) {
    for (const v of niceTicks(min, max, step)) {
      s.grid.push({ x1: plot.left, y1: sy(v), x2: plot.right, y2: sy(v) });
      s.ticks.push({ x: plot.left - 8, y: sy(v) + 3.5, text: tickText(data)(v), anchor: "end" });
    }
    const zeroY = sy(Math.max(min, Math.min(max, 0)));
    s.axis = `M${plot.left} ${plot.top}V${plot.bottom}M${plot.left} ${zeroY}H${plot.right}`;
  } else {
    for (const x of centers) s.grid.push({ x1: x, y1: plot.top - 10, x2: x, y2: plot.bottom });
  }
  centers.forEach((x, c) => s.categoryLabels.push({ x, y: plot.bottom + 20, lines: labelLines[c], fullText: categories[c], rotate: rotate ? -45 : undefined, anchor: rotate ? "end" : "middle" }));

  for (const t of data.annotations) {
    const y = sy(t.y);
    s.thresholds.push({
      d: `M${plot.left} ${y}H${plot.right}`,
      label: t.label ? { x: plot.right - 4, y: y - 6, text: t.label, anchor: "end" } : undefined,
    });
  }

  const n = data.series.length;
  const base = sy(Math.max(min, Math.min(max, 0)));

  if (type === "grouped-bars" || type === "lollipop") {
    const lollipop = type === "lollipop";
    const barW = lollipop ? 2 : Math.min(72, (colWidth * 0.74) / (n + 0.2 * (n - 1)));
    const gap = lollipop ? 0 : barW * 0.2;
    const slot = lollipop ? Math.min(26, (colWidth * 0.7) / n) : barW + gap;
    const span = lollipop ? slot * (n - 1) : n * barW + (n - 1) * gap;
    data.x.categories.forEach((_, c) => {
      data.series.forEach((ser, si) => {
        const i = c * n + si;
        const v = ser.y[c];
        const x = centers[c] - span / 2 + si * slot;
        const y = sy(v);
        const top = Math.min(y, base);
        s.barGroups.push({ i, from: "bottom", bars: [{ x: lollipop ? x - barW / 2 : x, y: top, w: barW, h: Math.abs(base - y), color: ser.color }] });
        const cx = lollipop ? x : x + barW / 2;
        if (lollipop) s.dots.push({ x: cx, y, color: ser.color, i });
        s.labels.push({ x: cx, y: (v >= 0 ? y : y + 14) - (lollipop ? 10 : 8), text: fmt(v), color: ser.color, anchor: "middle", i });
      });
    });
  } else if (type === "stacked-bars") {
    const barW = Math.min(80, colWidth * 0.56);
    data.x.categories.forEach((_, c) => {
      let pos = 0;
      let neg = 0;
      const bars = data.series.map(ser => {
        const v = ser.y[c];
        const from = v >= 0 ? pos : neg;
        const to = from + v;
        if (v >= 0) pos = to;
        else neg = to;
        return { x: centers[c] - barW / 2, y: sy(Math.max(from, to)), w: barW, h: Math.abs(sy(from) - sy(to)), color: ser.color };
      });
      s.barGroups.push({ i: c, from: "bottom", bars });
      s.labels.push({ x: centers[c], y: sy(pos) - 8, text: fmt(pos), color: "ink", anchor: "middle", i: c });
    });
  } else if (type === "waterfall") {
    const ser = data.series[0];
    const totals = runningTotals(ser.y);
    const barW = Math.min(64, colWidth * 0.6);
    const total = totals[totals.length - 1];
    categories.forEach((_, c) => {
      const isEnd = c === 0 || c === categories.length - 1;
      const from = isEnd ? 0 : totals[c - 1];
      const to = c === categories.length - 1 ? total : totals[c];
      const delta = to - from;
      const color: ChartColor = isEnd ? ser.color : delta >= 0 ? "teal" : "ember";
      s.barGroups.push({ i: c, from: "bottom", bars: [{ x: centers[c] - barW / 2, y: sy(Math.max(from, to)), w: barW, h: Math.abs(sy(from) - sy(to)), color }] });
      if (c > 0 && c < categories.length - 1) {
        s.grid.push({ x1: centers[c] + barW / 2, y1: sy(to), x2: centers[c + 1] - barW / 2, y2: sy(to) });
      }
      const text = isEnd ? fmt(to) : `${delta >= 0 ? "+" : "−"}${fmt(Math.abs(delta))}`;
      s.labels.push({ x: centers[c], y: sy(Math.max(from, to)) - 8, text, color: isEnd ? "ink" : color, anchor: "middle", i: c });
    });
  } else if (type === "slope") {
    data.series.forEach((ser, i) => {
      const [y0, y1] = [sy(ser.y[0]), sy(ser.y[1])];
      s.lines.push({ d: `M${centers[0]} ${y0}L${centers[1]} ${y1}`, color: ser.color, i, width: 2 });
      s.dots.push({ x: centers[0], y: y0, color: ser.color, i }, { x: centers[1], y: y1, color: ser.color, i });
      s.labels.push(
        { x: centers[0] - 10, y: y0 + 4, text: fmt(ser.y[0]), color: ser.color, anchor: "end", i },
        { x: centers[1] + 10, y: y1 + 4, text: fmt(ser.y[1]), color: ser.color, anchor: "start", i }
      );
    });
  }
  return s;
};

const layoutRows = (type: CategoryType, data: CategoryData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const fmt = format(data);
  const { min, max, step } = valueDomain(type, data);
  const { labelArea: labelWidth, lines: rowLines, height: requiredHeight } = rowPlan(type, data, width);
  height = Math.max(height, requiredHeight);
  const plot = { left: labelWidth + 16, right: width - 44, top: data.annotations.some(a => a.label) ? 22 : 8, bottom: height - 26 };
  const rowH = (plot.bottom - plot.top) / data.x.categories.length;
  const sx = (v: number) => plot.left + ((v - min) / (max - min)) * (plot.right - plot.left);

  for (const v of niceTicks(min, max, step)) {
    s.grid.push({ x1: sx(v), y1: plot.top, x2: sx(v), y2: plot.bottom });
    s.ticks.push({ x: sx(v), y: plot.bottom + 16, text: tickText(data)(v), anchor: "middle" });
  }
  data.x.categories.forEach((_, r) => {
    const lines = rowLines[r];
    s.categoryLabels.push({ x: plot.left - 10, y: plot.top + rowH * (r + 0.5) + 4 - (lines.length - 1) * 7, lines, fullText: data.x.categories[r], anchor: "end" });
  });
  for (const t of data.annotations) {
    const x = sx(t.y);
    s.thresholds.push({
      d: `M${x} ${plot.top}V${plot.bottom}`,
      label: t.label ? { x, y: plot.top - 7, text: t.label, anchor: "middle" } : undefined,
    });
  }

  const n = data.series.length;
  const base = sx(Math.max(min, Math.min(max, 0)));
  if (type === "horizontal-bars") {
    s.axis = `M${base} ${plot.top}V${plot.bottom}`;
    const barH = Math.min(22, (rowH * 0.72) / (n + 0.25 * (n - 1)));
    const gap = barH * 0.25;
    data.x.categories.forEach((_, r) => {
      data.series.forEach((ser, si) => {
        const i = r * n + si;
        const v = ser.y[r];
        const y = plot.top + rowH * (r + 0.5) - (n * barH + (n - 1) * gap) / 2 + si * (barH + gap);
        const x = sx(v);
        s.barGroups.push({ i, from: "left", bars: [{ x: Math.min(x, base), y, w: Math.abs(x - base), h: barH, color: ser.color }] });
        s.labels.push({ x: Math.max(x, base) + 5, y: y + barH / 2 + 3.5, text: fmt(v), color: ser.color, anchor: "start", i });
      });
    });
  } else {
    const [a, b] = data.series;
    data.x.categories.forEach((_, r) => {
      const y = plot.top + rowH * (r + 0.5);
      const [xa, xb] = [sx(a.y[r]), sx(b.y[r])];
      s.lines.push({ d: `M${Math.min(xa, xb)} ${y}H${Math.max(xa, xb)}`, color: "muted", i: r, width: 3 });
      s.dots.push({ x: xa, y, color: a.color, i: r }, { x: xb, y, color: b.color, i: r });
      const [lo, hi] = a.y[r] <= b.y[r] ? [[xa, a], [xb, b]] as const : [[xb, b], [xa, a]] as const;
      s.labels.push(
        { x: lo[0] - 8, y: y + 3.5, text: fmt(lo[1].y[r]), color: lo[1].color, anchor: "end", i: r },
        { x: hi[0] + 8, y: y + 3.5, text: fmt(hi[1].y[r]), color: hi[1].color, anchor: "start", i: r }
      );
    });
  }
  return s;
};

const layoutRadar = (data: CategoryData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const { min, max, step } = valueDomain("radar", data);
  const k = data.x.categories.length;
  const cx = width / 2;
  const cy = height / 2 + 4;
  const radius = Math.min(width / 2 - Math.min(110, Math.max(...data.x.categories.map(c => c.length)) * 3.2 + 30), height / 2 - 28);
  const at = (c: number, v: number) => {
    const r = ((v - min) / (max - min)) * radius;
    const a = (2 * Math.PI * c) / k;
    return [cx + r * Math.sin(a), cy - r * Math.cos(a)] as const;
  };
  const ring = (v: number) => data.x.categories.map((_, c) => at(c, v).map(n => n.toFixed(1)).join(",")).join(" ");

  for (const v of niceTicks(min, max, step)) if (v > min) s.rings.push(ring(v));
  data.x.categories.forEach((label, c) => {
    const [x, y] = at(c, max);
    s.grid.push({ x1: cx, y1: cy, x2: x, y2: y });
    const a = (2 * Math.PI * c) / k;
    const sin = Math.sin(a);
    const [lx, ly] = [cx + (radius + 14) * sin, cy - (radius + 14) * Math.cos(a)];
    const lines = wrap(label, 110);
    s.categoryLabels.push({
      x: lx,
      y: ly + 4 - (Math.cos(a) > 0.3 ? (lines.length - 1) * 14 : Math.cos(a) < -0.3 ? -6 : (lines.length - 1) * 7),
      lines,
      fullText: label,
      anchor: Math.abs(sin) < 0.2 ? "middle" : sin > 0 ? "start" : "end",
    });
  });
  for (const t of data.annotations) {
    const pts = ring(t.y);
    s.thresholds.push({
      d: `M${pts.replace(/ /g, "L")}Z`,
      label: t.label ? { x: at(0, t.y)[0] + 5, y: at(0, t.y)[1] - 5, text: t.label, anchor: "start" } : undefined,
    });
  }
  data.series.forEach((ser, i) => {
    const pts = ser.y.map((v, c) => at(c, v));
    s.polygons.push({ points: pts.map(p => p.map(n => n.toFixed(1)).join(",")).join(" "), color: ser.color, i, outline: true });
    pts.forEach(([x, y]) => s.dots.push({ x, y, color: ser.color, i }));
  });
  return s;
};
