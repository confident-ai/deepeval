import type * as React from "react";
import type { FunctionData } from "@/lib/charts/schema";
import type { Canvas } from "./_scene";
import { layoutFunction } from "./_function";
import styles from "./styles.module.scss";

const n1 = (v: number) => Math.round(v * 10) / 10;
const color = (name: string) => ({ "--mark-color": `var(--chart-${name})` }) as React.CSSProperties;

export const FunctionPlot = ({
  spec,
  canvas,
  clipId,
  className,
}: {
  spec: FunctionData;
  canvas: Canvas;
  clipId: string;
  className?: string;
}): React.ReactElement => {
  const l = layoutFunction(spec, canvas);
  const { plot } = l;
  return (
    <svg
      className={[styles.plot, className].filter(Boolean).join(" ")}
      viewBox={`0 0 ${canvas.width} ${canvas.height}`}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={plot.left} y={plot.top} width={plot.right - plot.left} height={plot.bottom - plot.top} />
        </clipPath>
      </defs>

      {l.xTicks.map(t => (
        <g key={`x${t.label}`}>
          <line className={t.zero ? styles.gridZero : styles.gridSolid} x1={n1(t.x)} x2={n1(t.x)} y1={plot.top} y2={plot.bottom} />
          <text className={styles.tick} x={n1(t.x)} y={plot.bottom + 16} textAnchor="middle">
            {t.label}
          </text>
        </g>
      ))}
      {l.yTicks.map(t => (
        <g key={`y${t.label}`}>
          <line className={t.zero ? styles.gridZero : styles.gridSolid} x1={plot.left} x2={plot.right} y1={n1(t.y)} y2={n1(t.y)} />
          <text className={styles.tick} x={plot.left - 8} y={n1(t.y + 3.5)} textAnchor="end">
            {t.label}
          </text>
        </g>
      ))}

      <g clipPath={`url(#${clipId})`}>
        {l.regions.map((r, i) => (
          <polygon key={i} className={styles.region} points={r.points} style={{ ...color(r.color), fillOpacity: r.opacity }} />
        ))}
        {l.hlines.map(h => (
          <line key={h.y} className={styles.threshold} x1={plot.left} x2={plot.right} y1={n1(h.y)} y2={n1(h.y)} />
        ))}
        {l.vlines.map(v => (
          <line key={v.x} className={styles.vline} style={color(v.color)} x1={n1(v.x)} x2={n1(v.x)} y1={plot.top} y2={plot.bottom} />
        ))}
        {l.drops.map((d, i) => (
          <line key={i} className={styles.drop} style={color(d.color)} x1={n1(d.x)} x2={n1(d.x)} y1={n1(d.y1)} y2={n1(d.y2)} />
        ))}
        {l.curves.map((c, i) => (
          <path
            key={i}
            className={c.dashed ? styles.curveDashed : styles.curve}
            style={{ ...color(c.color), "--i": i } as React.CSSProperties}
            d={c.d}
            pathLength={c.dashed ? undefined : 1}
          />
        ))}
      </g>

      {l.axes === "arrows" ? (
        <g className={styles.arrowAxes}>
          <path d={`M${plot.left} ${plot.bottom + 10}H${plot.right}M${plot.right - 6} ${plot.bottom + 6}L${plot.right} ${plot.bottom + 10}L${plot.right - 6} ${plot.bottom + 14}`} />
          <path d={`M${plot.left - 10} ${plot.bottom}V${plot.top}M${plot.left - 14} ${plot.top + 6}L${plot.left - 10} ${plot.top}L${plot.left - 6} ${plot.top + 6}`} />
        </g>
      ) : null}

      {l.vlines.map(v =>
        v.label ? (
          <text
            key={`vl${v.x}`}
            className={`${styles.halo} ${styles.annotationLabel}`}
            style={color(v.color)}
            x={n1(v.labelEnd ? v.x - 6 : v.x + 6)}
            y={plot.top + 12}
            textAnchor={v.labelEnd ? "end" : "start"}
          >
            {v.label}
          </text>
        ) : null
      )}
      {l.hlines.map(h =>
        h.label ? (
          <text key={`hl${h.y}`} className={`${styles.halo} ${styles.thresholdLabel}`} x={plot.right - 4} y={n1(h.y - 6)} textAnchor="end">
            {h.label}
          </text>
        ) : null
      )}
      {l.regions.map((r, i) =>
        r.label ? (
          <text key={`rl${i}`} className={`${styles.halo} ${styles.regionLabel}`} x={n1(r.label.x)} y={n1(r.label.y)} textAnchor="middle">
            {r.label.text}
          </text>
        ) : null
      )}
      {l.points.map((p, i) => (
        <g key={`p${i}`} className={styles.point} style={color(p.color)}>
          <circle className={p.hollow ? styles.dotHollow : styles.dot} cx={n1(p.x)} cy={n1(p.y)} r={4.5} />
          {p.label ? (
            <text className={`${styles.halo} ${styles.pointLabel}`} x={n1(p.x)} y={n1(p.y - 11)} textAnchor="middle">
              {p.label}
            </text>
          ) : null}
        </g>
      ))}

      {l.xLabel ? (
        <text className={styles.axisLabel} x={n1(l.xLabel.x)} y={n1(l.xLabel.y)} textAnchor="middle">
          {l.xLabel.text}
        </text>
      ) : null}
      {l.yLabel ? (
        <text
          className={styles.axisLabel}
          transform={`translate(${n1(l.yLabel.x)} ${n1(l.yLabel.y)}) rotate(-90)`}
          textAnchor="middle"
        >
          {l.yLabel.text}
        </text>
      ) : null}
    </svg>
  );
};
