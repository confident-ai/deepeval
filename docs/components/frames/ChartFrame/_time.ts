import { niceStep } from "@/lib/charts/domain";
import { type ChartType, type Moment, momentValue, type TimeData } from "@/lib/charts/schema";
import { niceTicks } from "./_scale";
import { assignLanes, type Canvas, emptyScene, type Scene, textWidth } from "./_scene";

export type TimeType = Extract<ChartType, "gantt" | "timeline">;

const CHAR = 7.4;
const DAY = 86_400_000;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const usesDates = (data: TimeData) =>
  data.series.some(s => [...(s.spans ?? []).flat(), ...(s.events ?? []).map(e => e[0])].some(m => typeof m === "string"));

const momentText = (days: number, dates: boolean) => {
  if (!dates) return String(+days.toFixed(2));
  const d = new Date(days * DAY);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
};

const timeTicks = (lo: number, hi: number, dates: boolean): { at: number; text: string }[] => {
  if (!dates) return niceTicks(lo, hi, niceStep(hi - lo, 5)).map(v => ({ at: v, text: momentText(v, false) }));
  const span = hi - lo;
  if (span > 100) {
    const every = span > 400 ? 3 : span > 200 ? 2 : 1;
    const out: { at: number; text: string }[] = [];
    const d = new Date(lo * DAY);
    let m = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1));
    while (m.getTime() / DAY <= hi) {
      if (m.getUTCMonth() % every === 0) out.push({ at: m.getTime() / DAY, text: MONTHS[m.getUTCMonth()] });
      m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1));
    }
    return out;
  }
  const step = span > 50 ? 14 : span > 12 ? 7 : span > 6 ? 2 : 1;

  const first = step >= 7 ? Math.ceil((lo - 4) / 7) * 7 + 4 : Math.ceil(lo);
  const out: { at: number; text: string }[] = [];
  for (let t = first; t <= hi; t += step) out.push({ at: t, text: momentText(t, true) });
  return out;
};

export const layoutTime = (type: TimeType, data: TimeData, canvas: Canvas): Scene =>
  type === "gantt" ? layoutGantt(data, canvas) : layoutTimeline(data, canvas);

const domain = (data: TimeData) => {
  const all = data.series.flatMap(s => [...(s.spans ?? []).flat(), ...(s.events ?? []).map(e => e[0])]).map(momentValue);
  const lo = data.x.min !== undefined ? momentValue(data.x.min) : Math.min(...all);
  const hi = data.x.max !== undefined ? momentValue(data.x.max) : Math.max(...all);
  const pad = data.x.min === undefined && data.x.max === undefined ? (hi - lo || 1) * 0.04 : 0;
  return { lo: lo - pad, hi: hi + pad };
};

const GANTT_ROW = 52;
const LANE = 16;

const ganttRows = (data: TimeData, width: number) => {
  const { lo, hi } = domain(data);
  const labelW = Math.min(width * 0.34, Math.max(...data.series.map(ser => ser.label.length)) * CHAR);
  const left = labelW + 16;
  const right = width - 16;
  const sx = (m: Moment | number) => left + (((typeof m === "number" ? m : momentValue(m)) - lo) / (hi - lo)) * (right - left);
  const rows = data.series.map(ser => {
    const events = (ser.events ?? []).map(([at, label]) => ({ x: sx(at), label }));
    const lanes = assignLanes(events.map(e => ({ x: e.x, width: textWidth(e.label) })), left, right);
    const count = Math.max(1, ...lanes.map(l => l.lane + 1));
    return { events, lanes, height: GANTT_ROW + (count - 1) * LANE };
  });
  return { lo, hi, labelW, left, right, sx, rows };
};

export const ganttHeight = (data: TimeData, width: number) =>
  8 + ganttRows(data, width).rows.reduce((sum, r) => sum + r.height, 0) + 26 + (data.x.label ? 20 : 0);

const layoutGantt = (data: TimeData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const dates = usesDates(data);
  const { lo, hi, labelW, left, right, sx, rows } = ganttRows(data, width);
  const plot = { left, right, top: 8, bottom: height - 26 - (data.x.label ? 20 : 0) };

  for (const t of timeTicks(lo, hi, dates)) {
    s.grid.push({ x1: sx(t.at), y1: plot.top, x2: sx(t.at), y2: plot.bottom });
    s.ticks.push({ x: sx(t.at), y: plot.bottom + 16, text: t.text, anchor: "middle" });
  }
  s.axis = `M${plot.left} ${plot.bottom}H${plot.right}`;
  if (data.x.label) s.axisLabels.push({ x: (plot.left + plot.right) / 2, y: height - 6, text: data.x.label });

  let y = plot.top;
  data.series.forEach((ser, i) => {
    const row = rows[i];
    const barH = 22;
    const barY = y + row.height - barH - 6;
    s.categoryLabels.push({ x: labelW + 6, y: barY + barH / 2 + 4.5, lines: [ser.label], anchor: "end" });
    s.barGroups.push({
      i,
      from: "left",
      bars: (ser.spans ?? []).map(([from, to]) => ({ x: sx(from), y: barY, w: Math.max(2, sx(to) - sx(from)), h: barH, color: ser.color })),
    });
    row.events.forEach((e, k) => {
      const { lane, anchor } = row.lanes[k];
      s.dots.push({ x: e.x, y: barY + barH / 2, color: ser.color, i, r: 4 });
      s.labels.push({ x: e.x, y: barY - 6 - lane * LANE, text: e.label, color: ser.color, anchor, i });
    });
    y += row.height;
  });
  return s;
};

const layoutTimeline = (data: TimeData, { width, height }: Canvas): Scene => {
  const s = emptyScene();
  const dates = usesDates(data);
  const { lo, hi } = domain(data);
  const plot = { left: 24, right: width - 24, top: 12, bottom: height - 24 - (data.x.label ? 20 : 0) };
  const axisY = (plot.top + plot.bottom) / 2;
  const sx = (m: Moment) => plot.left + ((momentValue(m) - lo) / (hi - lo)) * (plot.right - plot.left);

  for (const t of timeTicks(lo, hi, dates)) {
    const x = plot.left + ((t.at - lo) / (hi - lo)) * (plot.right - plot.left);
    s.grid.push({ x1: x, y1: plot.top, x2: x, y2: plot.bottom });
    s.ticks.push({ x, y: plot.bottom + 16, text: t.text, anchor: "middle" });
  }
  s.axis = `M${plot.left} ${axisY}H${plot.right}`;
  if (data.x.label) s.axisLabels.push({ x: (plot.left + plot.right) / 2, y: height - 6, text: data.x.label });

  data.series.forEach((ser, i) => {
    s.barGroups.push({
      i,
      from: "left",
      bars: (ser.spans ?? []).map(([from, to]) => ({ x: sx(from), y: axisY - 4, w: Math.max(2, sx(to) - sx(from)), h: 8, color: ser.color })),
    });
  });

  const events = data.series.flatMap((ser, i) =>
    (ser.events ?? []).map(([at, label]) => ({ x: sx(at), label, date: momentText(momentValue(at), dates), color: ser.color, i }))
  );
  const lanes = assignLanes(
    events.map(e => ({ x: e.x, width: Math.max(textWidth(e.label), textWidth(e.date)) })),
    plot.left,
    plot.right
  );
  const room = axisY - plot.top - 30;
  events.forEach((e, k) => {
    const { lane, anchor } = lanes[k];
    const up = lane % 2 === 0;
    const stem = Math.min(room, 24 + Math.floor(lane / 2) * 34);
    const end = up ? axisY - stem : axisY + stem;
    s.lines.push({ d: `M${e.x} ${axisY}V${end}`, color: "muted", i: e.i, width: 1 });
    s.dots.push({ x: e.x, y: axisY, color: e.color, i: e.i, r: 5 });
    s.labels.push({ x: e.x, y: up ? end - 18 : end + 14, text: e.label, color: "ink", anchor, i: e.i });
    s.labels.push({ x: e.x, y: up ? end - 5 : end + 27, text: e.date, color: "muted", anchor, i: e.i });
  });
  return s;
};
