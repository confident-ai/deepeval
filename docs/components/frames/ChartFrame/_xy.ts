import { labelWidth } from "./_labels";
import { xyDomain } from "@/lib/charts/domain";
import type { ChartType, XYData } from "@/lib/charts/schema";
import { niceTicks } from "./_scale";
import { type Canvas, emptyScene, linePath, pointList, type Scene } from "./_scene";

export type XYType = Extract<ChartType, "line" | "area" | "stacked-area" | "step" | "scatter" | "bubble" | "error-bars" | "band">;

const MAX_BUBBLE = 22;

export const layoutXY = (type: XYType, data: XYData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const { x: xd, y: yd } = xyDomain(type, data);
  const tickWidth = Math.max(...niceTicks(yd.min, yd.max, yd.step).map(v => labelWidth(`${+v.toFixed(6)}${data.y.unit}`, 11)));
  const left = Math.min(width * 0.35, Math.max(52, tickWidth + 12)) + (data.y.label ? 18 : 0);
  const bottom = 28 + (data.x.label ? 22 : 0);
  const plot = { left, right: width - 14, top: 16, bottom: height - bottom };
  const sx = (v: number) => plot.left + ((v - xd.min) / (xd.max - xd.min)) * (plot.right - plot.left);
  const sy = (v: number) => plot.bottom - ((v - yd.min) / (yd.max - yd.min)) * (plot.bottom - plot.top);
  const at = ([x, y]: readonly [number, number]) => [sx(x), sy(y)] as const;
  const tick = (axis: XYData["x"]) => (v: number) => `${+v.toFixed(6)}${axis.unit}`;

  for (const v of niceTicks(yd.min, yd.max, yd.step)) {
    s.grid.push({ x1: plot.left, y1: sy(v), x2: plot.right, y2: sy(v) });
    s.ticks.push({ x: plot.left - 8, y: sy(v) + 3.5, text: tick(data.y)(v), anchor: "end" });
  }
  for (const v of niceTicks(xd.min, xd.max, xd.step)) {
    s.ticks.push({ x: sx(v), y: plot.bottom + 17, text: tick(data.x)(v), anchor: "middle" });
  }
  s.axis = `M${plot.left} ${plot.top}V${plot.bottom}H${plot.right}`;
  if (data.x.label) s.axisLabels.push({ x: (plot.left + plot.right) / 2, y: height - 6, text: data.x.label });
  if (data.y.label) s.axisLabels.push({ x: 13, y: (plot.top + plot.bottom) / 2, text: data.y.label, rotate: true });

  for (const t of data.annotations) {
    const y = sy(t.y);
    s.thresholds.push({
      d: `M${plot.left} ${y}H${plot.right}`,
      label: t.label ? { x: plot.right - 4, y: y - 6, text: t.label, anchor: "end" } : undefined,
    });
  }

  const base = sy(Math.max(yd.min, Math.min(yd.max, 0)));

  if (type === "stacked-area") {
    let below: (readonly [number, number])[] = data.series[0].points.map(([x]) => [x, 0] as const);
    data.series.forEach((ser, i) => {
      const top = ser.points.map(([x, y], j) => [x, below[j][1] + y] as const);
      s.polygons.push({ points: pointList([...top.map(at), ...below.map(at).reverse()]), color: ser.color, i, outline: false, opacity: 0.45 });
      s.lines.push({ d: linePath(top.map(at)), color: ser.color, i, width: 1.5 });
      below = top;
    });
    return s;
  }

  const sizes = data.series.flatMap(ser => ser.size ?? []);
  const radius = (v: number) => Math.max(4, MAX_BUBBLE * Math.sqrt(v / Math.max(...sizes)));

  data.series.forEach((ser, i) => {
    const pts = ser.points.map(at);
    const err = (j: number) => ser.error?.[j] ?? 0;
    switch (type) {
      case "line":
      case "step":
        s.lines.push({ d: linePath(pts, type === "step"), color: ser.color, i, width: 2 });
        s.dots.push({ x: pts[pts.length - 1][0], y: pts[pts.length - 1][1], color: ser.color, i, r: 3.5 });
        break;
      case "area": {
        const first = pts[0][0];
        const last = pts[pts.length - 1][0];
        s.polygons.push({ points: pointList([...pts, [last, base], [first, base]]), color: ser.color, i, outline: false, opacity: 0.16 });
        s.lines.push({ d: linePath(pts), color: ser.color, i, width: 2 });
        break;
      }
      case "band": {
        const upper = ser.points.map(([x, y], j) => at([x, y + err(j)]));
        const lower = ser.points.map(([x, y], j) => at([x, y - err(j)]));
        s.polygons.push({ points: pointList([...upper, ...lower.reverse()]), color: ser.color, i, outline: false, opacity: 0.18 });
        s.lines.push({ d: linePath(pts), color: ser.color, i, width: 2 });
        break;
      }
      case "error-bars":
        s.lines.push({ d: linePath(pts), color: ser.color, i, width: 1.25 });
        ser.points.forEach(([x, y], j) => {
          const [px, top] = at([x, y + err(j)]);
          const bot = sy(y - err(j));
          s.lines.push({ d: `M${px - 4} ${top}H${px + 4}M${px} ${top}V${bot}M${px - 4} ${bot}H${px + 4}`, color: ser.color, i, width: 1.25 });
          s.dots.push({ x: px, y: sy(y), color: ser.color, i, r: 3.5 });
        });
        break;
      case "scatter":
        pts.forEach(([x, y]) => s.dots.push({ x, y, color: ser.color, i, r: 3.5 }));
        break;
      case "bubble":
        pts.forEach(([x, y], j) => s.dots.push({ x, y, color: ser.color, i, r: radius(ser.size?.[j] ?? 1), bubble: true }));
        break;
    }
  });
  return s;
};
