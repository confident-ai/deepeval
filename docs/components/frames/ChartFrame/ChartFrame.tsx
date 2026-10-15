import type * as React from "react";
import type { ChartDataMap } from "@/lib/charts/store";
import { type ChartData, isChartType, typeProblems } from "@/lib/charts/schema";
import FrameShell, { type LegendEntry } from "@/components/frames/FrameShell/FrameShell";
import { type CategoryType, layoutCategories, categoryCanvas } from "./_categories";
import type { Canvas } from "./_scene";
import { ScenePlot } from "./_ScenePlot";
import { layoutXY, type XYType } from "./_xy";
import { type DistributionType, layoutDistribution } from "./_distribution";
import { layoutParts, type PartsType } from "./_parts";
import { ganttHeight, layoutTime, type TimeType } from "./_time";
import { layoutMatrix, matrixCanvas, type MatrixType } from "./_matrix";
import { ChartReveal } from "./_ChartReveal";
import { FunctionPlot } from "./_FunctionPlot";
import styles from "./styles.module.scss";

interface ChartFrameProps {
  id: string;
  chart?: ChartData;

  type: string;
  title: string;

  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

const canvases = (type: string, chart: ChartData): [Canvas, Canvas] => {
  if (type === "gantt" && chart.shape === "time") {
    return [
      { width: 640, height: ganttHeight(chart.data, 640) },
      { width: 360, height: ganttHeight(chart.data, 360) },
    ];
  }
  if (chart.shape === "matrix") {
    return [matrixCanvas(type as MatrixType, chart.data, 640), matrixCanvas(type as MatrixType, chart.data, 360)];
  }
  if (type === "radar") return [{ width: 640, height: 380 }, { width: 360, height: 340 }];
  if (chart.shape === "categories") {
    return [categoryCanvas(type as CategoryType, chart.data, 640), categoryCanvas(type as CategoryType, chart.data, 360)];
  }
  return [{ width: 640, height: 320 }, { width: 360, height: 320 }];
};

const LINE_TYPES = new Set(["slope", "function", "line", "area", "stacked-area", "step", "error-bars", "band", "cdf"]);

const legendEntries = (type: string, chart: ChartData): LegendEntry[] => {
  const line = LINE_TYPES.has(type);
  if (chart.shape === "matrix") {
    // Heatmaps print their numbers; confusion matrices say what their two colors mean.
    return type === "confusion-matrix"
      ? [
          { id: "_agree", label: "agree", color: "teal" },
          { id: "_disagree", label: "disagree", color: "pink" },
        ]
      : [];
  }
  const entries: LegendEntry[] = [];

  for (const s of type === "funnel" || type === "gantt" ? [] : chart.data.series) {
    if (s.label) entries.push({ id: s.id, label: s.label, color: s.color, line, dashed: "dashed" in s && s.dashed });
  }

  if (chart.shape === "parts" && type !== "funnel") {
    const total = chart.data.series.reduce((sum, s) => sum + s.value, 0);
    for (const [k, s] of chart.data.series.entries()) {
      const share = total ? s.value / total : 0;
      entries[k].label = `${s.label} ${share > 0 && share < 0.01 ? "<1%" : `${Math.round(share * 100)}%`}`;
    }
  }
  if (type === "waterfall") {
    entries.push({ id: "_up", label: "increase", color: "teal", line }, { id: "_down", label: "decrease", color: "ember", line });
  }
  return entries;
};

const Drawing = ({ id, type, chart }: { id: string; type: string; chart: ChartData }): React.ReactElement => {
  const [wide, narrow] = canvases(type, chart);
  if (chart.shape === "function") {
    return (
      <>
        <FunctionPlot spec={chart.data} canvas={wide} clipId={`chart-${id}-wide`} className={styles.wide} />
        <FunctionPlot spec={chart.data} canvas={narrow} clipId={`chart-${id}-narrow`} className={styles.narrow} />
      </>
    );
  }
  const layout = (canvas: Canvas) =>
    chart.shape === "xy"
      ? layoutXY(type as XYType, chart.data, canvas)
      : chart.shape === "distribution"
        ? layoutDistribution(type as DistributionType, chart.data, canvas)
        : chart.shape === "parts"
          ? layoutParts(type as PartsType, chart.data, canvas)
          : chart.shape === "time"
            ? layoutTime(type as TimeType, chart.data, canvas)
            : chart.shape === "matrix"
              ? layoutMatrix(type as MatrixType, chart.data, canvas)
              : layoutCategories(type as CategoryType, chart.data, canvas);
  return (
    <>
      <ScenePlot scene={layout(wide)} canvas={wide} className={styles.wide} />
      <ScenePlot scene={layout(narrow)} canvas={narrow} className={styles.narrow} />
    </>
  );
};

const ChartFrame: React.FC<ChartFrameProps> = ({ id, chart, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !chart ? ["unknown or invalid data file"] : !isChartType(type) ? [`unknown type "${type}"`] : typeProblems(type, chart);
  if (!chart || problems.length) {
    console.warn(`[ChartFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }

  return (
    <FrameShell
      className={styles.frame}
      data={{ "data-chart": id, "data-chart-type": type }}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      legend={legendEntries(type, chart)}
      note={note}
    >
      <ChartReveal>
        <Drawing id={id} type={type} chart={chart} />
      </ChartReveal>
    </FrameShell>
  );
};

/** Bind MDX to this article's data; authors only supply the local ID. */
export const createChartFrame = (data: ChartDataMap) =>
  function ArticleChartFrame(props: Omit<ChartFrameProps, "chart">) {
    return <ChartFrame {...props} chart={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default ChartFrame;
