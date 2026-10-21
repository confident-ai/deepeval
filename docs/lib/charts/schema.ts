import { z } from "zod";
import { expressionError } from "@/lib/charts/expression";
import { distributionExtent, valueExtent, xyExtent } from "./domain";

export const CHART_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const CHART_COLORS = ["violet", "ember", "blue", "teal", "pink", "yellow", "muted", "ink"] as const;
export const chartColor = z.enum(CHART_COLORS);
export type ChartColor = z.infer<typeof chartColor>;

const nonEmpty = z.string().trim().min(1);

const seriesBase = {
  id: nonEmpty,
  label: nonEmpty,
  color: chartColor,
};

const thresholdAnnotation = z.strictObject({
  type: z.literal("threshold"),
  y: z.number(),
  label: nonEmpty.optional(),
});

const valueAxis = z.strictObject({
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().positive().optional(),
  unit: z.string().default(""),
  decimals: z.number().int().min(0).max(4).default(1),
});

export const categoryData = z.strictObject({
  x: z.strictObject({ categories: z.array(nonEmpty).min(1) }),
  y: valueAxis.default({ unit: "", decimals: 1 }),
  series: z
    .array(
      z.strictObject({
        ...seriesBase,

        y: z.array(z.number()).min(1),
      })
    )
    .min(1),
  annotations: z.array(z.discriminatedUnion("type", [thresholdAnnotation])).default([]),
});

export type CategoryData = z.infer<typeof categoryData>;

const rangeAxis = z.strictObject({
  min: z.number(),
  max: z.number(),
  step: z.number().positive().optional(),
  label: nonEmpty.optional(),
  decimals: z.number().int().min(0).max(4).default(1),
});

const curveRef = z.union([nonEmpty, z.number()]);

const vlineAnnotation = z.strictObject({
  type: z.literal("vline"),
  x: z.number(),
  label: nonEmpty.optional(),
  color: chartColor.default("muted"),
});

const regionAnnotation = z.strictObject({
  type: z.literal("region"),
  upper: curveRef,
  lower: curveRef,
  color: chartColor,
  opacity: z.number().min(0).max(1).default(0.14),
  label: nonEmpty.optional(),
  labelAt: z.number().optional(),
  from: z.number().optional(),
  to: z.number().optional(),
});

const pointsAnnotation = z.strictObject({
  type: z.literal("points"),
  series: nonEmpty,
  at: z.union([
    z.enum(["minima", "maxima", "roots"]),
    z.strictObject({ crosses: curveRef }),
    z.array(z.number()).min(1),
  ]),
  color: chartColor.optional(),
  hollow: z.boolean().default(false),
  label: nonEmpty.optional(),
  drop: z.union([curveRef, z.literal("axis")]).optional(),
});

const asymptotesAnnotation = z.strictObject({
  type: z.literal("asymptotes"),
  series: nonEmpty,
  label: nonEmpty.optional(),
});

export const functionData = z.strictObject({
  axes: z.enum(["grid", "arrows"]).default("grid"),
  x: rangeAxis,
  y: rangeAxis,
  series: z
    .array(
      z.strictObject({
        ...seriesBase,

        label: nonEmpty.optional(),
        y: z.union([nonEmpty, z.number()]),
        dashed: z.boolean().default(false),
        from: z.number().optional(),
        to: z.number().optional(),
      })
    )
    .min(1),
  annotations: z
    .array(
      z.discriminatedUnion("type", [thresholdAnnotation, vlineAnnotation, regionAnnotation, pointsAnnotation, asymptotesAnnotation])
    )
    .default([]),
});

export type FunctionData = z.infer<typeof functionData>;

const numberAxis = z.strictObject({
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().positive().optional(),
  label: nonEmpty.optional(),
  unit: z.string().default(""),
  decimals: z.number().int().min(0).max(4).default(1),
});

const defaultAxis = { unit: "", decimals: 1 };

export const xyData = z.strictObject({
  x: numberAxis.default(defaultAxis),
  y: numberAxis.default(defaultAxis),
  series: z
    .array(
      z.strictObject({
        ...seriesBase,
        points: z.array(z.tuple([z.number(), z.number()])).min(1),

        error: z.array(z.number().min(0)).optional(),
        size: z.array(z.number().positive()).optional(),
      })
    )
    .min(1),
  annotations: z.array(z.discriminatedUnion("type", [thresholdAnnotation])).default([]),
});

export type XYData = z.infer<typeof xyData>;

export const distributionData = z.strictObject({
  value: numberAxis.default(defaultAxis),
  series: z
    .array(
      z.strictObject({
        ...seriesBase,
        values: z.array(z.number()).min(1),
      })
    )
    .min(1),
  annotations: z
    .array(
      z.discriminatedUnion("type", [
        z.strictObject({ type: z.literal("threshold"), value: z.number(), label: nonEmpty.optional() }),
      ])
    )
    .default([]),
});

export type DistributionData = z.infer<typeof distributionData>;

export const partsData = z.strictObject({
  value: z
    .strictObject({
      unit: z.string().default(""),
      decimals: z.number().int().min(0).max(4).default(0),
    })
    .default({ unit: "", decimals: 0 }),
  series: z
    .array(
      z.strictObject({
        ...seriesBase,
        value: z.number().min(0),
      })
    )
    .min(1),
});

export type PartsData = z.infer<typeof partsData>;

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
const moment = z.union([
  z.number(),
  z
    .string()
    .regex(ISO_DATE, "use an ISO date like 2026-09-14")

    .refine(v => !Number.isNaN(Date.parse(v)) && new Date(`${v}T00:00:00Z`).toISOString().startsWith(v), "isn't a real date"),
]);
export type Moment = z.infer<typeof moment>;

export const timeData = z.strictObject({
  x: z
    .strictObject({
      label: nonEmpty.optional(),
      min: moment.optional(),
      max: moment.optional(),
    })
    .default({}),
  series: z
    .array(
      z.strictObject({
        ...seriesBase,

        spans: z.array(z.tuple([moment, moment])).optional(),

        events: z.array(z.tuple([moment, nonEmpty])).optional(),
      })
    )
    .min(1),
});

export type TimeData = z.infer<typeof timeData>;

/**
 * A grid of values: one row per y category, one column per x category. For a
 * confusion matrix, rows are the truth (y) and columns the prediction (x).
 */
export const matrixData = z.strictObject({
  x: z.strictObject({ label: nonEmpty.optional(), categories: z.array(nonEmpty).min(1) }),
  y: z.strictObject({ label: nonEmpty.optional(), categories: z.array(nonEmpty).min(1) }),
  values: z.array(z.array(z.number())).min(1),
  value: z
    .strictObject({
      unit: z.string().default(""),
      decimals: z.number().int().min(0).max(4).default(0),
    })
    .default({ unit: "", decimals: 0 }),
  color: chartColor.default("violet"),
});

export type MatrixData = z.infer<typeof matrixData>;

export const momentValue = (m: Moment): number => (typeof m === "number" ? m : Date.parse(`${m}T00:00:00Z`) / 86_400_000);

export type ChartData =
  | { shape: "categories"; data: CategoryData }
  | { shape: "xy"; data: XYData }
  | { shape: "distribution"; data: DistributionData }
  | { shape: "parts"; data: PartsData }
  | { shape: "time"; data: TimeData }
  | { shape: "function"; data: FunctionData }
  | { shape: "matrix"; data: MatrixData };
export type DataShape = ChartData["shape"];

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;

export const shapeOf = (raw: unknown): DataShape => {
  if (isObject(raw) && Array.isArray(raw.values)) return "matrix";
  if (isObject(raw) && isObject(raw.x) && "categories" in raw.x) return "categories";
  const seriesHave = (field: string) =>
    isObject(raw) && Array.isArray(raw.series) && raw.series.some(s => isObject(s) && field in s);
  if (seriesHave("points")) return "xy";
  if (seriesHave("values")) return "distribution";
  if (seriesHave("value")) return "parts";
  if (seriesHave("spans") || seriesHave("events")) return "time";
  return "function";
};

export const parseChartData = (raw: unknown) => {
  const shape = shapeOf(raw);
  if (shape === "categories") return { shape, result: categoryData.safeParse(raw) } as const;
  if (shape === "xy") return { shape, result: xyData.safeParse(raw) } as const;
  if (shape === "distribution") return { shape, result: distributionData.safeParse(raw) } as const;
  if (shape === "parts") return { shape, result: partsData.safeParse(raw) } as const;
  if (shape === "time") return { shape, result: timeData.safeParse(raw) } as const;
  if (shape === "matrix") return { shape, result: matrixData.safeParse(raw) } as const;
  return { shape, result: functionData.safeParse(raw) } as const;
};

type TypeRule =
  | { shape: "categories"; check?: (d: CategoryData) => string[] }
  | { shape: "xy"; check?: (d: XYData) => string[] }
  | { shape: "distribution"; check?: (d: DistributionData) => string[] }
  | { shape: "parts"; check?: (d: PartsData) => string[] }
  | { shape: "time"; check?: (d: TimeData) => string[] }
  | { shape: "function" }
  | { shape: "matrix"; check?: (d: MatrixData) => string[] };

const sliceCount = (type: string, max: number) => (d: PartsData) =>
  d.series.length <= max ? [] : [`${type} gets hard to read past ${max} slices, this data has ${d.series.length}`];

const valueCount = (type: string, min: number, max = Infinity) => (d: DistributionData) =>
  d.series.flatMap(s =>
    s.values.length < min
      ? [`${type} needs at least ${min} values per series, "${s.id}" has ${s.values.length}`]
      : s.values.length > max
        ? [`${type} gets unreadable past ${max} values per series, "${s.id}" has ${s.values.length}`]
        : []
  );

const seriesCount = (n: number, type: string) => (d: CategoryData) =>
  d.series.length === n ? [] : [`${type} needs exactly ${n} series, this data has ${d.series.length}`];

const increasingX = (type: string) => (d: XYData) =>
  d.series.flatMap(s =>
    s.points.every((p, i) => i === 0 || p[0] > s.points[i - 1][0])
      ? []
      : [`${type} joins points left to right, so "${s.id}" needs increasing x values`]
  );

const needsExtra = (field: "error" | "size", type: string) => (d: XYData) =>
  d.series.flatMap(s => (s[field] ? [] : [`${type} needs "${field}" on every series, "${s.id}" has none`]));

const all = (...checks: ((d: XYData) => string[])[]) => (d: XYData) => checks.flatMap(c => c(d));

export const CHART_TYPES = {
  "grouped-bars": { shape: "categories" },
  "stacked-bars": { shape: "categories" },
  "horizontal-bars": { shape: "categories" },
  lollipop: { shape: "categories" },
  dumbbell: { shape: "categories", check: seriesCount(2, "dumbbell") },
  slope: {
    shape: "categories",
    check: d => (d.x.categories.length === 2 ? [] : [`slope needs exactly 2 categories (before and after), this data has ${d.x.categories.length}`]),
  },
  radar: {
    shape: "categories",
    check: d => (d.x.categories.length >= 3 ? [] : [`radar needs at least 3 categories, this data has ${d.x.categories.length}`]),
  },

  waterfall: { shape: "categories", check: seriesCount(1, "waterfall") },

  line: { shape: "xy", check: increasingX("line") },
  area: { shape: "xy", check: increasingX("area") },
  "stacked-area": {
    shape: "xy",
    check: all(increasingX("stacked-area"), d => {
      const xs = (s: XYData["series"][number]) => s.points.map(p => p[0]).join(",");
      return d.series.every(s => xs(s) === xs(d.series[0])) ? [] : ["stacked-area needs every series to have the same x values"];
    }),
  },
  step: { shape: "xy", check: increasingX("step") },
  scatter: { shape: "xy" },
  bubble: { shape: "xy", check: needsExtra("size", "bubble") },
  "error-bars": { shape: "xy", check: all(increasingX("error-bars"), needsExtra("error", "error-bars")) },
  band: { shape: "xy", check: all(increasingX("band"), needsExtra("error", "band")) },

  histogram: { shape: "distribution", check: valueCount("histogram", 10) },

  box: { shape: "distribution", check: valueCount("box", 5) },
  strip: { shape: "distribution", check: valueCount("strip", 1, 300) },
  cdf: { shape: "distribution", check: valueCount("cdf", 5) },

  violin: { shape: "distribution", check: valueCount("violin", 20) },

  pie: { shape: "parts", check: sliceCount("pie", 6) },
  donut: { shape: "parts", check: sliceCount("donut", 6) },
  "stacked-100": { shape: "parts", check: sliceCount("stacked-100", 8) },
  waffle: { shape: "parts", check: sliceCount("waffle", 6) },
  treemap: { shape: "parts" },

  funnel: {
    shape: "parts",
    check: d =>
      d.series.flatMap((s, i) =>
        i > 0 && s.value > d.series[i - 1].value
          ? [`funnel stages narrow, so "${s.id}" (${s.value}) can't be larger than "${d.series[i - 1].id}" (${d.series[i - 1].value})`]
          : []
      ),
  },

  gantt: {
    shape: "time",
    check: d => d.series.flatMap(s => (s.spans?.length ? [] : [`gantt draws a row of spans per series, "${s.id}" has none`])),
  },
  timeline: {
    shape: "time",
    check: d => (d.series.some(s => s.events?.length) ? [] : ["timeline needs at least one event"]),
  },

  function: { shape: "function" },

  heatmap: { shape: "matrix" },
  "confusion-matrix": {
    shape: "matrix",
    check: d => {
      const same = d.x.categories.length === d.y.categories.length && d.x.categories.every((c, i) => c === d.y.categories[i]);
      const problems = same ? [] : ["confusion-matrix needs the same categories, in the same order, on x (predicted) and y (actual)"];
      if (d.values.flat().some(v => v < 0 || !Number.isInteger(v))) problems.push("confusion-matrix counts must be whole numbers, 0 or more");
      return problems;
    },
  },
} satisfies Record<string, TypeRule>;

export type ChartType = keyof typeof CHART_TYPES;
export const CHART_TYPE_NAMES = Object.keys(CHART_TYPES) as ChartType[];
export const isChartType = (type: string): type is ChartType => Object.hasOwn(CHART_TYPES, type);

export const dataProblems = (chart: ChartData): string[] => {
  const problems: string[] = [];
  if (chart.shape === "matrix") {
    const { data } = chart;
    if (data.values.length !== data.y.categories.length)
      problems.push(`"values" has ${data.values.length} rows but "y.categories" has ${data.y.categories.length}`);
    data.values.forEach((row, i) => {
      if (row.length !== data.x.categories.length)
        problems.push(`"values.${i}" has ${row.length} values but "x.categories" has ${data.x.categories.length}`);
    });
    return problems;
  }
  const ids = chart.data.series.map(s => s.id);
  if (new Set(ids).size !== ids.length) problems.push("series ids must be unique");
  const range = (axis: { min?: number; max?: number }, name: string) => {
    if (axis.min !== undefined && axis.max !== undefined && axis.max <= axis.min)
      problems.push(`"${name}.max" must be greater than "${name}.min"`);
  };

  if (chart.shape === "categories") {
    const { data } = chart;
    const count = data.x.categories.length;
    data.series.forEach((s, i) => {
      if (s.y.length !== count) problems.push(`"series.${i}.y" has ${s.y.length} values but "x.categories" has ${count}`);
    });
    range(data.y, "y");
  } else if (chart.shape === "xy") {
    const { data } = chart;
    range(data.x, "x");
    range(data.y, "y");
    data.series.forEach((s, i) => {
      for (const field of ["error", "size"] as const) {
        const extra = s[field];
        if (extra && extra.length !== s.points.length)
          problems.push(`"series.${i}.${field}" has ${extra.length} values but "series.${i}.points" has ${s.points.length}`);
      }
    });
  } else if (chart.shape === "distribution") {
    range(chart.data.value, "value");
  } else if (chart.shape === "time") {
    const { data } = chart;
    const moments = [
      ...data.series.flatMap(s => [...(s.spans ?? []).flat(), ...(s.events ?? []).map(e => e[0])]),
      ...[data.x.min, data.x.max].filter(m => m !== undefined),
    ];
    if (!moments.length) problems.push("needs at least one span or event");
    if (new Set(moments.map(m => typeof m)).size > 1) problems.push("use numbers or ISO dates throughout, not both");
    data.series.forEach((s, i) =>
      (s.spans ?? []).forEach(([from, to], j) => {
        if (momentValue(to) < momentValue(from)) problems.push(`"series.${i}.spans.${j}" ends (${to}) before it starts (${from})`);
      })
    );
    if (data.x.min !== undefined && data.x.max !== undefined && momentValue(data.x.max) <= momentValue(data.x.min))
      problems.push(`"x.max" must be after "x.min"`);
  } else if (chart.shape === "parts") {
    if (!chart.data.series.some(s => s.value > 0)) problems.push("at least one series needs a value above 0");
  } else {
    const { data } = chart;
    range(data.x, "x");
    range(data.y, "y");
    const known = new Set(ids);

    const checkRef = (ref: string | number, at: string) => {
      if (typeof ref !== "string" || known.has(ref)) return;
      const error = expressionError(ref);
      if (error) problems.push(`${at} is neither a series id nor an equation: "${ref}" (${error})`);
    };
    data.series.forEach((s, i) => {
      if (typeof s.y === "string") {
        const error = expressionError(s.y);
        if (error) problems.push(`"series.${i}.y" can't be read: ${error}`);
      }
    });
    data.annotations.forEach((a, i) => {
      const at = (field: string) => `"annotations.${i}.${field}"`;
      if (a.type === "region") {
        checkRef(a.upper, at("upper"));
        checkRef(a.lower, at("lower"));
      } else if (a.type === "asymptotes") {
        if (!known.has(a.series)) problems.push(`${at("series")} names unknown series "${a.series}"`);
      } else if (a.type === "points") {
        if (!known.has(a.series)) problems.push(`${at("series")} names unknown series "${a.series}"`);
        if (typeof a.at === "object" && !Array.isArray(a.at)) checkRef(a.at.crosses, at("at.crosses"));
        if (a.drop !== undefined && a.drop !== "axis") checkRef(a.drop, at("drop"));
      }
    });
  }
  return problems;
};

export const typeProblems = (type: ChartType, chart: ChartData): string[] => {
  const rule: TypeRule = CHART_TYPES[type];
  if (rule.shape !== chart.shape)
    return [`${type} draws ${rule.shape} data, but this file holds ${chart.shape} data`];
  if (rule.shape === "categories" && chart.shape === "categories") {
    const problems = rule.check ? rule.check(chart.data) : [];

    const { y } = chart.data;
    if (!problems.length && (y.min !== undefined || y.max !== undefined)) {
      const { lo, hi } = valueExtent(type, chart.data);
      if (y.min !== undefined && lo < y.min) problems.push(`${type} needs y down to ${lo}, but "y.min" is ${y.min}`);
      if (y.max !== undefined && hi > y.max) problems.push(`${type} needs y up to ${hi}, but "y.max" is ${y.max}`);
    }
    return problems;
  }
  if (rule.shape === "xy" && chart.shape === "xy") {
    const problems = rule.check ? rule.check(chart.data) : [];
    const { x, y } = chart.data;
    if (!problems.length) {
      const e = xyExtent(type, chart.data);
      if (x.min !== undefined && e.x.lo < x.min) problems.push(`${type} needs x down to ${e.x.lo}, but "x.min" is ${x.min}`);
      if (x.max !== undefined && e.x.hi > x.max) problems.push(`${type} needs x up to ${e.x.hi}, but "x.max" is ${x.max}`);
      if (y.min !== undefined && e.y.lo < y.min) problems.push(`${type} needs y down to ${e.y.lo}, but "y.min" is ${y.min}`);
      if (y.max !== undefined && e.y.hi > y.max) problems.push(`${type} needs y up to ${e.y.hi}, but "y.max" is ${y.max}`);
    }
    return problems;
  }
  if (rule.shape === "distribution" && chart.shape === "distribution") {
    const problems = rule.check ? rule.check(chart.data) : [];
    const { value } = chart.data;
    if (!problems.length) {
      const e = distributionExtent(chart.data);
      if (value.min !== undefined && e.lo < value.min) problems.push(`${type} needs values down to ${e.lo}, but "value.min" is ${value.min}`);
      if (value.max !== undefined && e.hi > value.max) problems.push(`${type} needs values up to ${e.hi}, but "value.max" is ${value.max}`);
    }
    return problems;
  }
  if (rule.shape === "parts" && chart.shape === "parts") return rule.check ? rule.check(chart.data) : [];
  if (rule.shape === "time" && chart.shape === "time") return rule.check ? rule.check(chart.data) : [];
  if (rule.shape === "matrix" && chart.shape === "matrix") return rule.check ? rule.check(chart.data) : [];
  return [];
};
