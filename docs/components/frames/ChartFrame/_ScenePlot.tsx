import { fitTicks, fitLabel } from "./_labels";
import type * as React from "react";
import type { Canvas, Scene } from "./_scene";
import styles from "./styles.module.scss";

const n1 = (v: number) => Math.round(v * 10) / 10;
const color = (name: string) => `var(--chart-${name})`;
const mark = (name: string, i: number) => ({ "--mark-color": color(name), "--i": i }) as React.CSSProperties;

export const ScenePlot = ({
  scene: s,
  canvas,
  className,
}: {
  scene: Scene;
  canvas: Canvas;
  className?: string;
}): React.ReactElement => (
  <svg
    className={[styles.plot, className].filter(Boolean).join(" ")}
    viewBox={`0 0 ${canvas.width} ${canvas.height}`}
    style={canvas.minWidth ? { minWidth: canvas.minWidth } : undefined}
    aria-hidden="true"
    focusable="false"
  >
    {s.rings.map((points, i) => (
      <polygon key={`r${i}`} className={styles.ring} points={points} />
    ))}
    {s.grid.map((g, i) => (
      <line key={`g${i}`} className={styles.grid} x1={n1(g.x1)} y1={n1(g.y1)} x2={n1(g.x2)} y2={n1(g.y2)} />
    ))}
    {fitTicks(s.ticks, canvas, s.axisLabels.some(a => a.rotate) ? 28 : 4).map(t => (
      <text key={`t${t.text}${t.x}${t.y}`} className={styles.tick} x={n1(t.x)} y={n1(t.y)} textAnchor={t.anchor}>
        <title>{t.fullText ?? t.text}</title>
        {t.text}
      </text>
    ))}
    {s.axisLabels.map(a => (
      <text
        key={`a${a.text}`}
        className={styles.axisLabel}
        x={a.rotate ? undefined : n1(a.x)}
        y={a.rotate ? undefined : n1(a.y)}
        transform={a.rotate ? `translate(${n1(a.x)} ${n1(a.y)}) rotate(-90)` : undefined}
        textAnchor="middle"
      >
        <title>{a.text}</title>
        {fitLabel(a.text, a.rotate ? canvas.height - 60 : canvas.width - 80, 12)}
      </text>
    ))}

    {s.barGroups.map(g => (
      <g
        key={`b${g.i}`}
        className={g.from === "left" ? styles.barBodyX : styles.barBody}
        style={{ "--i": g.i } as React.CSSProperties}
      >
        {g.bars.map((b, j) => (
          <rect
            key={j}
            className={b.solid ? styles.barSolid : styles.bar}
            style={{ "--mark-color": color(b.color) } as React.CSSProperties}
            x={n1(b.x)}
            y={n1(b.y)}
            width={n1(Math.max(0, b.w))}
            height={n1(Math.max(0, b.h))}
          />
        ))}
      </g>
    ))}
    {s.cells.map(c => (
      <rect
        key={`m${c.i}`}
        className={styles.cell}
        style={{ ...mark(c.color, c.i), fillOpacity: c.opacity }}
        x={n1(c.x)}
        y={n1(c.y)}
        width={n1(c.w)}
        height={n1(c.h)}
      />
    ))}
    {s.polygons.map((p, j) => (
      <polygon
        key={`p${j}`}
        className={p.slice ? styles.slice : p.outline ? styles.area : styles.areaFill}
        style={{ ...mark(p.color, p.i), ...(p.opacity !== undefined ? { fillOpacity: p.opacity } : {}) }}
        points={p.points}
      />
    ))}
    {s.lines.map((l, j) => (
      <path key={`l${j}`} className={styles.curve} style={{ ...mark(l.color, l.i), strokeWidth: l.width }} d={l.d} pathLength={1} />
    ))}
    {s.axis ? <path className={styles.axis} d={s.axis} /> : null}
    {s.thresholds.map((t, j) => (
      <g key={`th${j}`}>
        <path className={styles.threshold} d={t.d} fill="none" />
        {t.label ? (
          <text className={`${styles.halo} ${styles.thresholdLabel}`} x={n1(t.label.x)} y={n1(t.label.y)} textAnchor={t.label.anchor}>
            {t.label.text}
          </text>
        ) : null}
      </g>
    ))}
    {s.dots.map((d, j) => (
      <circle
        key={`d${j}`}
        className={`${styles.point} ${d.bubble ? styles.bubble : styles.dot}`}
        style={mark(d.color, d.i)}
        cx={n1(d.x)}
        cy={n1(d.y)}
        r={n1(d.r ?? 4.5)}
      />
    ))}
    {s.labels.map((l, j) => (
      <text
        key={`v${j}`}
        className={[l.plain ? "" : styles.halo, styles.value, l.large ? styles.valueLarge : "", l.small ? styles.valueSmall : ""].filter(Boolean).join(" ")}
        style={mark(l.color, l.i)}
        x={n1(l.x)}
        y={n1(l.y)}
        textAnchor={l.anchor}
      >
        {l.text}
      </text>
    ))}
    {s.categoryLabels.map((c, j) => (
      <text key={`c${j}`} className={styles.category} x={n1(c.x)} y={n1(c.y)} textAnchor={c.anchor} transform={c.rotate ? `rotate(${c.rotate} ${n1(c.x)} ${n1(c.y)})` : undefined}>
        <title>{c.fullText ?? c.lines.join(" ")}</title>
        {c.lines.map((line, k) => (
          <tspan key={k} x={n1(c.x)} dy={k ? 15 : 0}>
            {line}
          </tspan>
        ))}
      </text>
    ))}
  </svg>
);
