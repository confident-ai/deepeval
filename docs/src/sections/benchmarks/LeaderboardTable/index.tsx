"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import {
  formatDate,
  formatMetric,
  scoreFraction,
} from "@/lib/benchmarks/format";
import type { BenchmarkMetric, RankedEntry } from "@/lib/benchmarks/types";
import ProviderMark from "../ProviderMark";
import { RankBadge } from "../RankTrophy";
import styles from "./LeaderboardTable.module.scss";

type LeaderboardTableProps = {
  entries: RankedEntry[];
  metrics: BenchmarkMetric[];
  primaryMetricId: string;
};

type SortState = {
  metricId: string;
  // "desc" puts the largest value first.
  order: "asc" | "desc";
};

const bestFirstOrder = (metric: BenchmarkMetric): SortState["order"] =>
  metric.higherIsBetter ? "desc" : "asc";

const LeaderboardTable: React.FC<LeaderboardTableProps> = ({
  entries,
  metrics,
  primaryMetricId,
}) => {
  const primary = metrics.find((metric) => metric.id === primaryMetricId)!;
  const [sort, setSort] = useState<SortState>({
    metricId: primary.id,
    order: bestFirstOrder(primary),
  });

  const sorted = useMemo(() => {
    const sign = sort.order === "desc" ? -1 : 1;
    return [...entries].sort(
      (a, b) => sign * (a.scores[sort.metricId] - b.scores[sort.metricId]),
    );
  }, [entries, sort]);

  const primaryValues = useMemo(
    () => entries.map((entry) => entry.scores[primary.id]),
    [entries, primary.id],
  );

  const onSort = (metric: BenchmarkMetric) => {
    setSort((current) =>
      current.metricId === metric.id
        ? { metricId: metric.id, order: current.order === "desc" ? "asc" : "desc" }
        : { metricId: metric.id, order: bestFirstOrder(metric) },
    );
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.rankCol}>
                #
              </th>
              <th scope="col">Model</th>
              {metrics.map((metric) => {
                const active = sort.metricId === metric.id;
                const Icon = sort.order === "desc" ? ArrowDown : ArrowUp;
                return (
                  <th
                    key={metric.id}
                    scope="col"
                    className={styles.numCol}
                    aria-sort={
                      active
                        ? sort.order === "desc"
                          ? "descending"
                          : "ascending"
                        : "none"
                    }
                  >
                    <button
                      type="button"
                      className={styles.sortButton}
                      data-active={active}
                      title={metric.description}
                      onClick={() => onSort(metric)}
                    >
                      {metric.label}
                      <Icon className={styles.sortIcon} aria-hidden />
                    </button>
                  </th>
                );
              })}
              <th scope="col" className={styles.dateCol}>
                Evaluated
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((entry) => (
              <tr key={entry.model} data-top={entry.rank === 1}>
                <td className={styles.rankCol}>
                  <RankBadge rank={entry.rank} />
                </td>
                <th scope="row" className={styles.modelCell}>
                  <span className={styles.modelRow}>
                    <ProviderMark provider={entry.provider} />
                    <span>
                      <span className={styles.model}>
                        {entry.model}
                        {entry.openWeights && (
                          <span className={styles.tag}>Open weights</span>
                        )}
                      </span>
                      <span className={styles.provider}>{entry.provider}</span>
                    </span>
                  </span>
                </th>
                {metrics.map((metric) => {
                  const value = entry.scores[metric.id];
                  const isPrimary = metric.id === primary.id;
                  return (
                    <td key={metric.id} className={styles.numCol}>
                      <span className={styles.value} data-primary={isPrimary}>
                        {formatMetric(value, metric)}
                        {isPrimary && entry.confidenceInterval !== undefined && (
                          <span className={styles.interval}>
                            ±{entry.confidenceInterval.toFixed(metric.decimals ?? 1)}
                          </span>
                        )}
                      </span>
                      {isPrimary && (
                        <span className={styles.track} aria-hidden>
                          <span
                            className={styles.fill}
                            style={{
                              width: `${scoreFraction(value, metric, primaryValues) * 100}%`,
                            }}
                          />
                        </span>
                      )}
                    </td>
                  );
                })}
                <td className={styles.dateCol}>
                  {formatDate(entry.evaluatedAt)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default LeaderboardTable;
