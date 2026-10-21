import type { CategoryData, ChartType, DistributionData, XYData } from "./schema";

export interface Extent {
  lo: number;
  hi: number;

  zero: boolean;
}

export const runningTotals = (values: number[]): number[] => {
  const out: number[] = [];
  values.forEach((v, i) => out.push(i === 0 ? v : out[i - 1] + v));
  return out;
};

const withZero = ({ lo, hi, zero }: Extent): Extent =>
  zero ? { lo: Math.min(0, lo), hi: Math.max(0, hi), zero } : { lo, hi, zero };

export const valueExtent = (type: ChartType, data: CategoryData): Extent => {
  const all = data.series.flatMap(s => s.y);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  let zero = true;
  if (type === "stacked-bars") {
    const sums = data.x.categories.map((_, c) => data.series.map(s => s.y[c]));
    lo = Math.min(0, ...sums.map(v => v.filter(n => n < 0).reduce((a, b) => a + b, 0)));
    hi = Math.max(...sums.map(v => v.filter(n => n > 0).reduce((a, b) => a + b, 0)));
  } else if (type === "waterfall") {
    const totals = runningTotals(data.series[0]?.y ?? []);
    lo = Math.min(...totals);
    hi = Math.max(...totals);
  } else if (type === "dumbbell" || type === "slope") {
    zero = false;
  }
  const thresholds = data.annotations.map(a => a.y);
  return withZero({ lo: Math.min(lo, ...thresholds), hi: Math.max(hi, ...thresholds), zero });
};

export const xyExtent = (type: ChartType, data: XYData): { x: Extent; y: Extent } => {
  const points = data.series.flatMap(s => s.points);
  const xs = points.map(p => p[0]);
  let ys = points.map(p => p[1]);
  if (type === "stacked-area") {
    ys = data.series[0].points.map((_, i) => data.series.reduce((sum, s) => sum + s.points[i][1], 0));
  } else if (type === "error-bars" || type === "band") {
    ys = data.series.flatMap(s => s.points.flatMap((p, i) => [p[1] - (s.error?.[i] ?? 0), p[1] + (s.error?.[i] ?? 0)]));
  }
  const thresholds = data.annotations.map(a => a.y);
  return {
    x: { lo: Math.min(...xs), hi: Math.max(...xs), zero: false },
    y: withZero({
      lo: Math.min(...ys, ...thresholds),
      hi: Math.max(...ys, ...thresholds),
      zero: type === "area" || type === "stacked-area",
    }),
  };
};

export const distributionExtent = (data: DistributionData): Extent => {
  const values = [...data.series.flatMap(s => s.values), ...data.annotations.map(a => a.value)];
  return { lo: Math.min(...values), hi: Math.max(...values), zero: false };
};

export const niceStep = (span: number, target = 4) => {
  const raw = span / target || 1;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
};

export const niceRange = (
  extent: Extent,
  axis: { min?: number; max?: number; step?: number },
  tight = false
) => {
  const lo = axis.min ?? extent.lo;
  const hi = axis.max ?? extent.hi;
  const pad = tight || (extent.zero && axis.min === undefined) ? 0 : (hi - lo || 1) * 0.06;
  const step = axis.step ?? niceStep(hi - lo + 2 * pad);

  const padded = Math.floor((lo - pad) / step + 1e-9) * step;
  const min = axis.min ?? (tight ? lo : lo >= 0 && padded < 0 ? 0 : padded);
  const max = axis.max ?? (tight ? hi : Math.ceil((hi + pad) / step - 1e-9) * step);
  return { min, max: max > min ? max : min + step, step };
};

export const valueDomain = (type: ChartType, data: CategoryData) => niceRange(valueExtent(type, data), data.y);

export const xyDomain = (type: ChartType, data: XYData) => {
  const extent = xyExtent(type, data);

  const tightX = type !== "scatter" && type !== "bubble";
  return { x: niceRange(extent.x, data.x, tightX), y: niceRange(extent.y, data.y) };
};

export const distributionDomain = (data: DistributionData) => niceRange(distributionExtent(data), data.value);
