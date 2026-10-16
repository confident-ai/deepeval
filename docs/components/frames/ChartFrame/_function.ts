import { asymptotes, crossings, maxima, minima, roots, safe } from "@/lib/charts/analysis";
import { compileExpression, type Fn } from "@/lib/charts/expression";
import type { ChartColor, FunctionData } from "@/lib/charts/schema";
import type { Canvas } from "./_scene";
import { niceTicks } from "./_scale";

export interface FunctionLayout {
  plot: { left: number; right: number; top: number; bottom: number };
  axes: FunctionData["axes"];
  xTicks: { x: number; label: string; zero: boolean }[];
  yTicks: { y: number; label: string; zero: boolean }[];
  xLabel?: { x: number; y: number; text: string };
  yLabel?: { x: number; y: number; text: string };
  regions: { points: string; color: ChartColor; opacity: number; label?: { x: number; y: number; text: string } }[];
  curves: { d: string; color: ChartColor; dashed: boolean }[];
  hlines: { y: number; label?: string }[];
  vlines: { x: number; color: ChartColor; label?: string; labelEnd: boolean }[];
  drops: { x: number; y1: number; y2: number; color: ChartColor }[];
  points: { x: number; y: number; color: ChartColor; hollow: boolean; label?: string }[];
}

const n1 = (v: number) => Math.round(v * 10) / 10;
const tickLabel = (v: number) => (Math.abs(v) < 1e-10 ? "0" : String(+v.toPrecision(4)).replace("-", "−"));

export const layoutFunction = (spec: FunctionData, { width, height }: Canvas): FunctionLayout => {
  const grid = spec.axes === "grid";
  const left = (grid ? 40 : 28) + (spec.y.label ? 20 : 0);
  const bottom = (grid ? 26 : 22) + (spec.x.label ? 22 : 0);
  const plot = { left, right: width - 12, top: 14, bottom: height - bottom };
  const { min: x0, max: x1 } = spec.x;
  const { min: y0, max: y1 } = spec.y;
  const span = y1 - y0;
  const sx = (x: number) => plot.left + ((x - x0) / (x1 - x0)) * (plot.right - plot.left);
  const sy = (y: number) => plot.bottom - ((y - y0) / span) * (plot.bottom - plot.top);
  const inX = (x: number) => x >= x0 - 1e-9 && x <= x1 + 1e-9;
  const inY = (y: number) => y >= y0 - 1e-9 && y <= y1 + 1e-9;
  const clampY = (y: number) => Math.min(y1, Math.max(y0, y));

  const fns = new Map<string, Fn>(
    spec.series.map(s => [s.id, typeof s.y === "number" ? () => s.y as number : compileExpression(s.y)])
  );

  const fnOf = (ref: string | number): Fn =>
    typeof ref === "number" ? () => ref : (fns.get(ref) ?? compileExpression(ref));
  const colorOf = new Map(spec.series.map(s => [s.id, s.color]));

  const xTicks = grid
    ? niceTicks(x0, x1, spec.x.step).map(v => ({ x: sx(v), label: tickLabel(v), zero: v === 0 }))
    : [];
  const yTicks = grid
    ? niceTicks(y0, y1, spec.y.step).map(v => ({ y: sy(v), label: tickLabel(v), zero: v === 0 }))
    : [];
  const xLabel = spec.x.label
    ? { x: (plot.left + plot.right) / 2, y: height - 6, text: spec.x.label }
    : undefined;
  const yLabel = spec.y.label
    ? { x: 12, y: (plot.top + plot.bottom) / 2, text: spec.y.label }
    : undefined;

  const curvePath = (f: Fn, a: number, b: number) => {
    const n = 900;
    const outside = (y: number) => y > y1 || y < y0;
    let d = "";
    let pen = false;
    let prev: number | null = null;
    for (let i = 0; i <= n; i++) {
      const x = a + ((b - a) * i) / n;
      const y = safe(f, x);
      if (!Number.isFinite(y)) {
        pen = false;
        prev = null;
        continue;
      }
      if (prev !== null && Math.abs(y - prev) > 2 * span && (outside(y) || outside(prev))) pen = false;
      const yc = Math.min(y1 + 10 * span, Math.max(y0 - 10 * span, y));
      d += `${pen ? "L" : "M"}${n1(sx(x))} ${n1(sy(yc))}`;
      pen = true;
      prev = y;
    }
    return d;
  };

  const regions = spec.annotations.flatMap(a => {
    if (a.type !== "region") return [];
    const f = fnOf(a.upper);
    const g = fnOf(a.lower);
    const from = Math.max(a.from ?? x0, x0);
    const to = Math.min(a.to ?? x1, x1);
    const n = 300;
    const top: string[] = [];
    const bot: string[] = [];
    for (let i = 0; i <= n; i++) {
      const x = from + ((to - from) * i) / n;
      const ya = safe(f, x);
      const yb = safe(g, x);
      if (!Number.isFinite(ya) || !Number.isFinite(yb)) continue;
      top.push(`${n1(sx(x))},${n1(sy(clampY(ya)))}`);
      bot.push(`${n1(sx(x))},${n1(sy(clampY(yb)))}`);
    }
    let label: { x: number; y: number; text: string } | undefined;
    if (a.label) {
      let at = a.labelAt;
      if (at === undefined) {
        let best = -1;
        const pad = (to - from) * 0.1;
        for (let i = 0; i <= 200; i++) {
          const x = from + pad + ((to - from - 2 * pad) * i) / 200;
          const h = Math.abs(clampY(safe(f, x)) - clampY(safe(g, x)));
          const score = h * (1 - (0.25 * Math.abs(i - 100)) / 100);
          if (Number.isFinite(score) && score > best) {
            best = score;
            at = x;
          }
        }
      }
      const ym = at === undefined ? NaN : (clampY(safe(f, at)) + clampY(safe(g, at))) / 2;
      if (at !== undefined && Number.isFinite(ym)) label = { x: sx(at), y: sy(ym) + 4, text: a.label };
    }
    return [{ points: top.concat(bot.reverse()).join(" "), color: a.color, opacity: a.opacity, label }];
  });

  const curves = spec.series.map(s => ({
    d: curvePath(fns.get(s.id)!, Math.max(s.from ?? x0, x0), Math.min(s.to ?? x1, x1)),
    color: s.color,
    dashed: s.dashed,
  }));

  const hlines = spec.annotations.flatMap(a =>
    a.type === "threshold" && inY(a.y) ? [{ y: sy(a.y), label: a.label }] : []
  );

  const vlines = spec.annotations.flatMap(a => {
    if (a.type === "vline")
      return inX(a.x) ? [{ x: sx(a.x), color: a.color, label: a.label, labelEnd: sx(a.x) > plot.right - 90 }] : [];
    if (a.type !== "asymptotes") return [];
    const label = a.label ?? "x = {x}";
    return asymptotes(fns.get(a.series)!, x0, x1, span).map(x => ({
      x: sx(x),
      color: "muted" as const,
      label: label.replace("{x}", x.toFixed(spec.x.decimals)),
      labelEnd: sx(x) > plot.right - 90,
    }));
  });

  const drops: FunctionLayout["drops"] = [];
  const points: FunctionLayout["points"] = [];
  for (const a of spec.annotations) {
    if (a.type !== "points") continue;
    const f = fns.get(a.series)!;
    const xs =
      a.at === "minima"
        ? minima(f, x0, x1, span)
        : a.at === "maxima"
          ? maxima(f, x0, x1, span)
          : a.at === "roots"
            ? roots(f, x0, x1, span)
            : Array.isArray(a.at)
              ? a.at
              : crossings(f, fnOf(a.at.crosses), x0, x1, span);
    const color = a.color ?? colorOf.get(a.series) ?? "muted";
    for (const x of xs) {
      const y = safe(f, x);
      if (!inX(x) || !Number.isFinite(y) || !inY(y)) continue;
      if (a.drop !== undefined) {
        const to = a.drop === "axis" ? y0 : safe(fnOf(a.drop), x);
        if (Number.isFinite(to)) drops.push({ x: sx(x), y1: sy(y), y2: sy(clampY(to)), color });
      }
      const label = a.label
        ?.replace("{x}", x.toFixed(spec.x.decimals))
        .replace("{y}", y.toFixed(spec.y.decimals));
      points.push({ x: sx(x), y: sy(y), color, hollow: a.hollow, label });
    }
  }

  return { plot, axes: spec.axes, xTicks, yTicks, xLabel, yLabel, regions, curves, hlines, vlines, drops, points };
};
