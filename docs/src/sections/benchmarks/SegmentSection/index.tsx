"use client";

import { useState } from "react";
import { ChartColumn, Grid3x3 } from "lucide-react";
import { getMetric, rankEntries } from "@/lib/benchmarks/format";
import type { Benchmark } from "@/lib/benchmarks/types";
import FilterPills from "../FilterPills";
import SegmentBars from "../SegmentBars";
import SegmentHeatmap from "../SegmentHeatmap";
import ViewToggle from "../ViewToggle";
import styles from "./SegmentSection.module.scss";

type SegmentSectionProps = {
  benchmark: Benchmark;
};

type View = "bars" | "heatmap";

// Grouped bars stay readable up to this many segments per model; beyond
// it the heatmap is the better default.
const MAX_BAR_SEGMENTS = 4;

const SegmentSection: React.FC<SegmentSectionProps> = ({ benchmark }) => {
  const [metricId, setMetricId] = useState(benchmark.primaryMetricId);
  const [view, setView] = useState<View>(
    benchmark.segments.length <= MAX_BAR_SEGMENTS ? "bars" : "heatmap",
  );

  const metric = getMetric(benchmark.metrics, metricId);

  return (
    <div className={styles.wrap}>
      <div className={styles.controls}>
        <FilterPills
          label="Metric"
          value={metricId}
          onChange={setMetricId}
          options={benchmark.metrics.map((item) => ({
            id: item.id,
            label: item.label,
          }))}
        />
        <ViewToggle<View>
          value={view}
          onChange={setView}
          options={[
            { id: "bars", label: "Bars", icon: <ChartColumn aria-hidden /> },
            { id: "heatmap", label: "Heatmap", icon: <Grid3x3 aria-hidden /> },
          ]}
        />
      </div>

      {view === "bars" ? (
        <SegmentBars
          entries={rankEntries(benchmark)}
          segments={benchmark.segments}
          metric={metric}
        />
      ) : (
        <SegmentHeatmap benchmark={benchmark} metric={metric} />
      )}
    </div>
  );
};

export default SegmentSection;
