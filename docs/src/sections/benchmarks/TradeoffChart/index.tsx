"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  formatMetric,
  paretoFront,
  type TradeoffPoint,
} from "@/lib/benchmarks/format";
import { niceStep } from "@/lib/benchmarks/scale";
import type { BenchmarkMetric } from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import ProviderMark from "../ProviderMark";
import { Trophy, hasTrophy } from "../RankTrophy";
import styles from "./TradeoffChart.module.scss";

type TradeoffChartProps = {
  points: TradeoffPoint[];
  xMetric: BenchmarkMetric;
  yMetric: BenchmarkMetric;
};

const MARGIN = { top: 40, right: 32, bottom: 44, left: 56 };
const MARKER = 28;
const LABEL_HEIGHT = 16;
const LABEL_GAP = 6;
// Rough advance width of one label character at 12px.
const CHAR_WIDTH = 6.4;
const TROPHY_WIDTH = 18;

// A spread this wide on a linear axis would pile most points into one corner.
const LOG_RATIO = 20;

function linearTicks(min: number, max: number): number[] {
  const step = niceStep((max - min) / 4 || 1);
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = start; value <= end + step / 2; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  const first = Math.floor(Math.log10(min));
  const last = Math.ceil(Math.log10(max));
  for (let exponent = first; exponent <= last; exponent += 1) {
    for (const unit of [1, 2, 5]) {
      ticks.push(unit * 10 ** exponent);
    }
  }
  const lower = ticks.filter((tick) => tick <= min).pop() ?? min;
  const upper = ticks.find((tick) => tick >= max) ?? max;
  return ticks.filter((tick) => tick >= lower && tick <= upper);
}

type Scale = {
  ticks: number[];
  position: (value: number) => number;
};

function buildScale(values: number[], length: number): Scale {
  const min = Math.min(...values);
  const max = Math.max(...values);

  if (min > 0 && max / min > LOG_RATIO) {
    const ticks = logTicks(min, max);
    const lo = Math.log10(ticks[0]);
    const hi = Math.log10(ticks[ticks.length - 1]);
    return {
      ticks,
      position: (value) => ((Math.log10(value) - lo) / (hi - lo)) * length,
    };
  }

  const ticks = linearTicks(min, max);
  const lo = ticks[0];
  const hi = ticks[ticks.length - 1];
  return {
    ticks,
    position: (value) => ((value - lo) / (hi - lo)) * length,
  };
}

type Box = { left: number; top: number; right: number; bottom: number };

const overlaps = (a: Box, b: Box) =>
  a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

type Placed = {
  point: TradeoffPoint;
  x: number;
  y: number;
  onFront: boolean;
  // Top-left of the label, or null when no free spot was found. The model
  // is still named in the tooltip.
  label: { left: number; top: number } | null;
};

const TradeoffChart: React.FC<TradeoffChartProps> = ({
  points,
  xMetric,
  yMetric,
}) => {
  const [active, setActive] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(760);

  // Drawn at the container's real pixel width so type and markers keep
  // their size instead of scaling with the chart.
  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const height = Math.round(Math.min(440, Math.max(300, width * 0.46)));
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const { xScale, yScale, placed, frontPath } = useMemo(() => {
    const xScale = buildScale(
      points.map((point) => point.x),
      plotW,
    );
    const yScale = buildScale(
      points.map((point) => point.y),
      plotH,
    );
    const px = (value: number) => MARGIN.left + xScale.position(value);
    const py = (value: number) => MARGIN.top + plotH - yScale.position(value);

    const front = paretoFront(
      points,
      xMetric.higherIsBetter,
      yMetric.higherIsBetter,
    );
    const frontNames = new Set(front.map((point) => point.model));

    const markers: Box[] = points.map((point) => ({
      left: px(point.x) - MARKER / 2,
      right: px(point.x) + MARKER / 2,
      top: py(point.y) - MARKER / 2,
      bottom: py(point.y) + MARKER / 2,
    }));
    const labels: Box[] = [];

    // Best-ranked models claim label space first.
    const order = [...points].sort(
      (a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity),
    );
    const placed: Placed[] = order.map((point) => {
      const x = px(point.x);
      const y = py(point.y);
      const labelWidth =
        point.model.length * CHAR_WIDTH +
        (point.rank !== undefined && hasTrophy(point.rank) ? TROPHY_WIDTH : 0);
      const reach = MARKER / 2 + LABEL_GAP;
      const candidates = [
        { left: x + reach, top: y - LABEL_HEIGHT / 2 },
        { left: x - reach - labelWidth, top: y - LABEL_HEIGHT / 2 },
        { left: x - labelWidth / 2, top: y - reach - LABEL_HEIGHT },
        { left: x - labelWidth / 2, top: y + reach },
      ];
      const spot = candidates.find(({ left, top }) => {
        const box: Box = {
          left,
          top,
          right: left + labelWidth,
          bottom: top + LABEL_HEIGHT,
        };
        const inside =
          box.left >= 0 &&
          box.right <= width &&
          box.top >= 0 &&
          box.bottom <= height - MARGIN.bottom + 4;
        const self = markers[points.indexOf(point)];
        return (
          inside &&
          !labels.some((other) => overlaps(box, other)) &&
          !markers.some((other) => other !== self && overlaps(box, other))
        );
      });
      if (spot) {
        labels.push({
          left: spot.left,
          top: spot.top,
          right: spot.left + labelWidth,
          bottom: spot.top + LABEL_HEIGHT,
        });
      }
      return {
        point,
        x,
        y,
        onFront: frontNames.has(point.model),
        label: spot ?? null,
      };
    });

    return {
      xScale,
      yScale,
      placed,
      frontPath: front.map((point) => `${px(point.x)},${py(point.y)}`).join(" "),
    };
  }, [points, plotW, plotH, width, height, xMetric.higherIsBetter, yMetric.higherIsBetter]);

  const px = (value: number) => MARGIN.left + xScale.position(value);
  const py = (value: number) => MARGIN.top + plotH - yScale.position(value);
  const activeItem = placed.find((item) => item.point.model === active);

  const tickLabel = (value: number, metric: BenchmarkMetric) =>
    formatMetric(value, {
      ...metric,
      decimals: value % 1 === 0 ? 0 : metric.decimals,
    });

  return (
    <figure className={styles.figure}>
      <div ref={wrapRef} className={styles.plot} style={{ height }}>
        <svg
          width={width}
          height={height}
          className={styles.svg}
          role="img"
          aria-label={`${yMetric.label} against ${xMetric.label} for ${points.length} models`}
        >
          {yScale.ticks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={py(tick)}
                y2={py(tick)}
                className={styles.grid}
              />
              <text
                x={MARGIN.left - 12}
                y={py(tick)}
                className={styles.tick}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {tickLabel(tick, yMetric)}
              </text>
            </g>
          ))}
          {xScale.ticks.map((tick) => (
            <text
              key={`x-${tick}`}
              x={px(tick)}
              y={height - MARGIN.bottom + 20}
              className={styles.tick}
              textAnchor="middle"
            >
              {tickLabel(tick, xMetric)}
            </text>
          ))}
          <polyline className={styles.front} points={frontPath} />
        </svg>

        <span className={styles.yTitle}>{yMetric.label}</span>
        <span className={styles.xTitle}>{xMetric.label}</span>

        {placed.map(({ point, x, y, onFront, label }) => (
          <div
            key={point.model}
            className={styles.item}
            data-front={onFront}
            data-active={active === point.model}
          >
            <button
              type="button"
              className={styles.marker}
              style={{ left: x, top: y }}
              aria-label={`${point.model}: ${formatMetric(point.y, yMetric)} ${yMetric.label}, ${formatMetric(point.x, xMetric)} ${xMetric.label}`}
              onMouseEnter={() => setActive(point.model)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(point.model)}
              onBlur={() => setActive(null)}
            >
              <ProviderMark provider={point.provider} size="sm" />
            </button>
            {label && (
              <span
                className={styles.label}
                style={{ left: label.left, top: label.top }}
                aria-hidden
              >
                {point.rank !== undefined && (
                  <Trophy rank={point.rank} size={13} />
                )}
                {point.model}
              </span>
            )}
          </div>
        ))}

        {activeItem && (
          <div
            className={styles.tooltip}
            style={{ left: activeItem.x, top: activeItem.y }}
            data-flip={activeItem.x > width * 0.6}
            role="status"
          >
            <strong>
              <ModelName
                model={activeItem.point.model}
                provider={activeItem.point.provider}
              />
            </strong>
            <dl>
              <div>
                <dt>{yMetric.label}</dt>
                <dd>{formatMetric(activeItem.point.y, yMetric)}</dd>
              </div>
              <div>
                <dt>{xMetric.label}</dt>
                <dd>{formatMetric(activeItem.point.x, xMetric)}</dd>
              </div>
            </dl>
          </div>
        )}
      </div>

      <figcaption className={styles.caption}>
        <span className={styles.legendItem}>
          <span className={styles.legendLine} aria-hidden />
          Best trade-offs
        </span>
        <span>
          {yMetric.higherIsBetter ? "Higher" : "Lower"} and further{" "}
          {xMetric.higherIsBetter ? "right" : "left"} is better
        </span>
      </figcaption>
    </figure>
  );
};

export default TradeoffChart;
