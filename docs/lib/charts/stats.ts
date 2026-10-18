const sorted = (values: number[]) => [...values].sort((a, b) => a - b);

export const quantile = (values: number[], p: number): number => {
  const s = sorted(values);
  const at = (s.length - 1) * p;
  const lo = Math.floor(at);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (at - lo);
};

export interface BoxStats {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;

  low: number;
  high: number;
  outliers: number[];
}

export const boxStats = (values: number[]): BoxStats => {
  const s = sorted(values);
  const q1 = quantile(s, 0.25);
  const q3 = quantile(s, 0.75);
  const reach = 1.5 * (q3 - q1);
  const inside = s.filter(v => v >= q1 - reach && v <= q3 + reach);
  return {
    n: s.length,
    min: s[0],
    q1,
    median: quantile(s, 0.5),
    q3,
    max: s[s.length - 1],
    low: inside[0] ?? q1,
    high: inside[inside.length - 1] ?? q3,
    outliers: s.filter(v => v < q1 - reach || v > q3 + reach),
  };
};

export const histogram = (values: number[], start: number, width: number, bins: number): number[] => {
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor((v - start) / width)))]++;
  return counts;
};

export const density = (values: number[], at: number[]): number[] => {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, n - 1));
  const iqr = quantile(values, 0.75) - quantile(values, 0.25);
  const spread = Math.min(sd, iqr / 1.34) || sd || 1;
  const h = 0.9 * spread * n ** -0.2;
  return at.map(x => values.reduce((sum, v) => sum + Math.exp(-0.5 * ((x - v) / h) ** 2), 0) / (n * h * Math.sqrt(2 * Math.PI)));
};
