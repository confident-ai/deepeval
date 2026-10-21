import type { ChartData } from "./schema";
import { CHART_KIND } from "@/lib/frames/kinds";
import { loadFrameFile } from "@/lib/frames/store";

export type ChartDataMap = Record<string, ChartData>;

/** Chart IDs are local to the charts.json beside the content file. */
export const loadChartData = (sourceFile: string): ChartDataMap => loadFrameFile(CHART_KIND, sourceFile);
