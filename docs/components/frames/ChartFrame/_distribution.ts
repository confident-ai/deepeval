import { distributionDomain, niceRange, niceStep } from "@/lib/charts/domain";
import type { ChartType, DistributionData } from "@/lib/charts/schema";
import { boxStats, density, histogram, quantile } from "@/lib/charts/stats";
import { niceTicks } from "./_scale";
import { type Canvas, emptyScene, linePath, pointList, type Scene } from "./_scene";

export type DistributionType = Extract<ChartType, "histogram" | "box" | "strip" | "cdf" | "violin">;

const CHAR = 7.4;

export const layoutDistribution = (type: DistributionType, data: DistributionData, canvas: Canvas): Scene =>
  type === "histogram" || type === "cdf" ? layoutAcross(type, data, canvas) : layoutColumns(type, data, canvas);

const fmt = (data: DistributionData) => (v: number) => `${+v.toFixed(6)}${data.value.unit}`;

const layoutAcross = (type: DistributionType, data: DistributionData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const vd = distributionDomain(data);
  const plot = { left: 58, right: width - 14, top: 16, bottom: height - 28 - (data.value.label ? 22 : 0) };
  const sx = (v: number) => plot.left + ((v - vd.min) / (vd.max - vd.min)) * (plot.right - plot.left);

  const most = Math.max(...data.series.map(ser => ser.values.length));
  const binWidth = niceStep(vd.max - vd.min, Math.min(20, Math.max(10, Math.round(Math.sqrt(most) * 1.5))));
  const binStart = Math.floor(vd.min / binWidth + 1e-9) * binWidth;
  const bins = Math.max(1, Math.ceil((vd.max - binStart) / binWidth - 1e-9));
  const counts = data.series.map(ser => histogram(ser.values, binStart, binWidth, bins));

  const yd =
    type === "histogram"
      ? niceRange({ lo: 0, hi: Math.max(...counts.flat()), zero: true }, {})
      : { min: 0, max: 1, step: 0.25 };
  const sy = (v: number) => plot.bottom - ((v - yd.min) / (yd.max - yd.min)) * (plot.bottom - plot.top);

  for (const v of niceTicks(yd.min, yd.max, yd.step)) {
    s.grid.push({ x1: plot.left, y1: sy(v), x2: plot.right, y2: sy(v) });
    s.ticks.push({ x: plot.left - 8, y: sy(v) + 3.5, text: type === "cdf" ? `${Math.round(v * 100)}%` : String(v), anchor: "end" });
  }
  for (const v of niceTicks(vd.min, vd.max, vd.step)) s.ticks.push({ x: sx(v), y: plot.bottom + 17, text: fmt(data)(v), anchor: "middle" });
  s.axis = `M${plot.left} ${plot.top}V${plot.bottom}H${plot.right}`;
  if (data.value.label) s.axisLabels.push({ x: (plot.left + plot.right) / 2, y: height - 6, text: data.value.label });
  s.axisLabels.push({ x: 13, y: (plot.top + plot.bottom) / 2, text: type === "cdf" ? "share at or below" : "count", rotate: true });

  for (const t of data.annotations) {
    const x = sx(t.value);
    s.thresholds.push({ d: `M${x} ${plot.top}V${plot.bottom}`, label: t.label ? { x: x + 5, y: plot.top + 10, text: t.label, anchor: "start" } : undefined });
  }

  data.series.forEach((ser, i) => {
    if (type === "histogram") {
      counts[i].forEach((count, b) => {
        if (!count) return;
        const x0 = sx(binStart + b * binWidth);
        const x1 = sx(binStart + (b + 1) * binWidth);
        s.barGroups.push({ i: i * bins + b, from: "bottom", bars: [{ x: x0 + 0.5, y: sy(count), w: x1 - x0 - 1, h: plot.bottom - sy(count), color: ser.color }] });
      });
    } else {
      const v = [...ser.values].sort((a, b) => a - b);
      const pts: [number, number][] = [[sx(vd.min), sy(0)]];
      v.forEach((value, j) => pts.push([sx(value), sy((j + 1) / v.length)]));
      pts.push([sx(vd.max), sy(1)]);
      s.lines.push({ d: linePath(pts, true), color: ser.color, i, width: 2 });
    }
  });
  return s;
};

const wrap = (text: string, width: number): string[] => {
  const max = Math.max(4, Math.floor(width / CHAR));
  if (text.length <= max) return [text];
  const lines: string[] = [""];
  for (const word of text.split(" ")) {
    const line = lines[lines.length - 1];
    if (!line || `${line} ${word}`.length <= max || lines.length === 2) lines[lines.length - 1] = line ? `${line} ${word}` : word;
    else lines.push(word);
  }
  return lines;
};

const jitter = (j: number) => ((j * 0.618034) % 1) - 0.5;

const layoutColumns = (type: DistributionType, data: DistributionData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const vd = distributionDomain(data);
  const left = 52 + (data.value.label ? 18 : 0);
  const colWidth = (width - left - 10) / data.series.length;
  const labelLines = data.series.map(ser => wrap(ser.label, colWidth - 6));
  const plot = { left, right: width - 10, top: 16, bottom: height - 14 - 15 * Math.max(...labelLines.map(l => l.length)) };
  const sy = (v: number) => plot.bottom - ((v - vd.min) / (vd.max - vd.min)) * (plot.bottom - plot.top);
  const half = Math.min(42, colWidth * 0.32);

  for (const v of niceTicks(vd.min, vd.max, vd.step)) {
    s.grid.push({ x1: plot.left, y1: sy(v), x2: plot.right, y2: sy(v) });
    s.ticks.push({ x: plot.left - 8, y: sy(v) + 3.5, text: fmt(data)(v), anchor: "end" });
  }
  s.axis = `M${plot.left} ${plot.top}V${plot.bottom}H${plot.right}`;
  if (data.value.label) s.axisLabels.push({ x: 13, y: (plot.top + plot.bottom) / 2, text: data.value.label, rotate: true });
  for (const t of data.annotations) {
    const y = sy(t.value);
    s.thresholds.push({ d: `M${plot.left} ${y}H${plot.right}`, label: t.label ? { x: plot.right - 4, y: y - 6, text: t.label, anchor: "end" } : undefined });
  }

  data.series.forEach((ser, i) => {
    const cx = plot.left + colWidth * (i + 0.5);
    s.categoryLabels.push({ x: cx, y: plot.bottom + 20, lines: labelLines[i], anchor: "middle" });
    const st = boxStats(ser.values);

    if (type === "box") {
      s.lines.push({ d: `M${cx} ${sy(st.high)}V${sy(st.q3)}M${cx} ${sy(st.q1)}V${sy(st.low)}M${cx - half / 2} ${sy(st.high)}H${cx + half / 2}M${cx - half / 2} ${sy(st.low)}H${cx + half / 2}`, color: ser.color, i, width: 1.25 });
      s.barGroups.push({ i, from: "bottom", bars: [{ x: cx - half, y: sy(st.q3), w: half * 2, h: sy(st.q1) - sy(st.q3), color: ser.color }] });
      s.lines.push({ d: `M${cx - half} ${sy(st.median)}H${cx + half}`, color: ser.color, i, width: 2.5 });
      st.outliers.forEach(v => s.dots.push({ x: cx, y: sy(v), color: ser.color, i, r: 2.5 }));
      s.labels.push({ x: cx + half + 6, y: sy(st.median) + 4, text: fmt(data)(+st.median.toFixed(data.value.decimals)), color: ser.color, anchor: "start", i });
    } else if (type === "strip") {
      [...ser.values].sort((a, b) => a - b).forEach((v, j) => s.dots.push({ x: cx + jitter(j) * half * 1.6, y: sy(v), color: ser.color, i, r: 2.5 }));
      s.lines.push({ d: `M${cx - half} ${sy(st.median)}H${cx + half}`, color: ser.color, i, width: 2 });
    } else {
      const steps = 64;
      const at = Array.from({ length: steps + 1 }, (_, k) => st.min + ((st.max - st.min) * k) / steps);
      const d = density(ser.values, at);
      const peak = Math.max(...d) || 1;
      const right = at.map((v, k) => [cx + (d[k] / peak) * half, sy(v)] as const);
      const leftSide = at.map((v, k) => [cx - (d[k] / peak) * half, sy(v)] as const).reverse();
      s.polygons.push({ points: pointList([...right, ...leftSide]), color: ser.color, i, outline: true });
      s.lines.push({ d: `M${cx} ${sy(quantile(ser.values, 0.75))}V${sy(quantile(ser.values, 0.25))}`, color: ser.color, i, width: 3 });
      s.dots.push({ x: cx, y: sy(st.median), color: ser.color, i, r: 3.5 });
    }
  });
  return s;
};
