"use client";

import { useState } from "react";
import { formatMetric } from "@/lib/benchmarks/format";
import { zeroBasedTicks } from "@/lib/benchmarks/scale";
import type {
  BenchmarkMetric,
  BenchmarkSegment,
  RankedEntry,
} from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import ProviderMark from "../ProviderMark";
import styles from "./SegmentBars.module.scss";

type SegmentBarsProps = {
  entries: RankedEntry[];
  segments: BenchmarkSegment[];
  metric: BenchmarkMetric;
};

// Above this many bars per model the value labels would collide, so they
// are left to the hover card.
const MAX_LABELLED_SEGMENTS = 2;

const SegmentBars: React.FC<SegmentBarsProps> = ({
  entries,
  segments,
  metric,
}) => {
  const [active, setActive] = useState<string | null>(null);

  const valueOf = (entry: RankedEntry, segment: BenchmarkSegment) =>
    entry.segmentScores[metric.id][segment.id];

  const max = Math.max(
    ...entries.flatMap((entry) =>
      segments.map((segment) => valueOf(entry, segment)),
    ),
  );
  const ticks = zeroBasedTicks(max);
  const top = ticks[ticks.length - 1];
  const labelled = segments.length <= MAX_LABELLED_SEGMENTS;

  const tickLabel = (value: number) =>
    formatMetric(value, {
      ...metric,
      decimals: value % 1 === 0 ? 0 : metric.decimals,
    });

  return (
    <figure className={styles.figure}>
      <ul className={styles.legend}>
        {segments.map((segment, index) => (
          <li key={segment.id}>
            <span className={styles.swatch} data-series={index} aria-hidden />
            {segment.label}
          </li>
        ))}
        <li className={styles.direction}>
          {metric.higherIsBetter ? "Higher" : "Lower"} is better
        </li>
      </ul>

      <div className={styles.scroll}>
        <div
          className={styles.chart}
          style={{ ["--bars" as string]: segments.length }}
        >
          <div className={styles.axis} aria-hidden>
            {ticks.map((tick) => (
              <span
                key={tick}
                className={styles.tick}
                style={{ bottom: `${(tick / top) * 100}%` }}
              >
                {tickLabel(tick)}
              </span>
            ))}
          </div>

          <div className={styles.groups}>
            <div className={styles.grid} aria-hidden>
              {ticks.map((tick) => (
                <span
                  key={tick}
                  className={styles.gridLine}
                  style={{ bottom: `${(tick / top) * 100}%` }}
                />
              ))}
            </div>

            {entries.map((entry, entryIndex) => {
              const values = segments.map((segment) => valueOf(entry, segment));
              const gap = Math.max(...values) - Math.min(...values);
              const edge =
                entryIndex < 2
                  ? "start"
                  : entryIndex > entries.length - 3
                    ? "end"
                    : "middle";
              return (
                <div
                  key={entry.model}
                  className={styles.group}
                  data-active={active === entry.model}
                  data-dimmed={active !== null && active !== entry.model}
                  tabIndex={0}
                  onMouseEnter={() => setActive(entry.model)}
                  onMouseLeave={() => setActive(null)}
                  onFocus={() => setActive(entry.model)}
                  onBlur={() => setActive(null)}
                  aria-label={`${entry.model}: ${segments
                    .map(
                      (segment, index) =>
                        `${segment.label} ${formatMetric(values[index], metric)}`,
                    )
                    .join(", ")}`}
                >
                  <div className={styles.bars}>
                    {segments.map((segment, index) => (
                      <span
                        key={segment.id}
                        className={styles.bar}
                        data-series={index}
                        style={{ height: `${(values[index] / top) * 100}%` }}
                      >
                        {labelled && (
                          <span className={styles.value}>
                            {formatMetric(values[index], metric)}
                          </span>
                        )}
                      </span>
                    ))}
                  </div>

                  <div className={styles.label}>
                    <ProviderMark provider={entry.provider} size="sm" />
                    <span className={styles.labelText}>{entry.model}</span>
                  </div>

                  {active === entry.model && (
                    <div className={styles.tooltip} data-edge={edge} role="status">
                      <strong>
                        <ModelName model={entry.model} provider={entry.provider} />
                      </strong>
                      <dl>
                        {segments.map((segment, index) => (
                          <div key={segment.id}>
                            <dt>
                              <span
                                className={styles.swatch}
                                data-series={index}
                                aria-hidden
                              />
                              {segment.label}
                            </dt>
                            <dd>{formatMetric(values[index], metric)}</dd>
                          </div>
                        ))}
                        <div className={styles.gapRow}>
                          <dt>Gap</dt>
                          <dd>{formatMetric(gap, metric)}</dd>
                        </div>
                      </dl>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </figure>
  );
};

export default SegmentBars;
