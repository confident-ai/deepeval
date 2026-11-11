import {
  formatCount,
  formatMetric,
  rankEntries,
} from "@/lib/benchmarks/format";
import type { Benchmark, BenchmarkMetric } from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import { Trophy } from "../RankTrophy";
import styles from "./SegmentHeatmap.module.scss";

type SegmentHeatmapProps = {
  benchmark: Benchmark;
  // The metric the cells show. Rows stay in overall rank order.
  metric: BenchmarkMetric;
};

const SegmentHeatmap: React.FC<SegmentHeatmapProps> = ({
  benchmark,
  metric: primary,
}) => {
  const ranked = rankEntries(benchmark);

  const all = ranked.flatMap((entry) =>
    benchmark.segments.map(
      (segment) => entry.segmentScores[primary.id][segment.id],
    ),
  );
  const min = Math.min(...all);
  const max = Math.max(...all);

  // 0 for the worst cell in the table, 1 for the best.
  const strength = (value: number) => {
    if (max === min) return 1;
    const t = (value - min) / (max - min);
    return primary.higherIsBetter ? t : 1 - t;
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" className={styles.num}>
                Overall
              </th>
              {benchmark.segments.map((segment) => (
                <th key={segment.id} scope="col" className={styles.num}>
                  {segment.label}
                  <span className={styles.count}>
                    {formatCount(segment.caseCount)} {benchmark.dataset.unit}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ranked.map((entry) => (
              <tr key={entry.model}>
                <th scope="row">
                  <ModelName model={entry.model} provider={entry.provider}>
                    <Trophy rank={entry.rank} size={14} />
                  </ModelName>
                </th>
                <td className={styles.num}>
                  {formatMetric(entry.scores[primary.id], primary)}
                </td>
                {benchmark.segments.map((segment) => {
                  const value = entry.segmentScores[primary.id][segment.id];
                  return (
                    <td
                      key={segment.id}
                      className={styles.cell}
                      style={{
                        ["--strength" as string]: strength(value).toFixed(3),
                      }}
                      title={`${entry.model}, ${segment.label}: ${formatMetric(value, primary)}`}
                    >
                      {formatMetric(value, primary)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.legend}>
        <span>Weaker</span>
        <span className={styles.ramp} aria-hidden />
        <span>Stronger</span>
      </p>
    </div>
  );
};

export default SegmentHeatmap;
