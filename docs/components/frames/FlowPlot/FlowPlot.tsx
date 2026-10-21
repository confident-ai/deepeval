import type * as React from "react";
import { markColor } from "@/components/frames/FrameShell/FrameShell";
import { FLOW_ICON_COMPONENTS } from "./_icons";
import { blockHeight, type FlowScene, GROUP_HEAD, ICON, iconRoom, LINE, type SceneNode, SMALL_LINE } from "./_scene";
import styles from "./styles.module.scss";

interface FlowPlotProps {
  scene: FlowScene;
  /** Unique within the page; names the arrowhead marker. */
  id: string;
  /** Wide shows above a 520px frame, narrow below it. */
  variant: "wide" | "narrow";
  /** How far the drawing may shrink to fit before it scrolls instead. */
  minScale: number;
}

const Lines = ({ x, y, lines, className, lineHeight, anchor }: { x: number; y: number; lines: string[]; className: string; lineHeight: number; anchor?: "start" | "middle" }): React.ReactElement => (
  <text className={className} x={x} y={y} textAnchor={anchor ?? "middle"} dominantBaseline="central">
    {lines.map((line, i) => (
      <tspan key={i} x={x} dy={i ? lineHeight : 0}>
        {line}
      </tspan>
    ))}
  </text>
);

// Average glyph width used to center an icon and its label as one unit; nearer
// the real font than the layout's deliberately roomy estimate.
const CENTERING_CHAR = 0.56;

/** A node's label (and smaller lines under it), with its icon to the left; centered in the node. */
const NodeContent = ({ node }: { node: SceneNode }): React.ReactElement | null => {
  if (node.ghost) return null;
  const sub = node.sub ?? [];
  const cx = node.x + node.w / 2;
  const cy = node.y + node.h / 2 + (node.shape === "store" ? 3 : 0);
  // First line's center, so the label and sub-lines together sit in the middle.
  const top = cy - blockHeight(node.lines.length, sub.length) / 2 + LINE / 2;
  const subTop = top + (node.lines.length - 1) * LINE + LINE / 2 + 2 + SMALL_LINE / 2;
  const Icon = node.icon ? FLOW_ICON_COMPONENTS[node.icon] : null;
  const textW = Math.max(...node.lines.map(l => Array.from(l).length * 13 * CENTERING_CHAR));
  const x0 = cx - (iconRoom(node.icon) + textW) / 2;
  const tx = Icon ? x0 + iconRoom(node.icon) : cx;
  return (
    <>
      {Icon ? (
        <Icon className={styles.icon} x={x0} y={top - ICON / 2} width={ICON} height={ICON} strokeWidth={1.75} aria-hidden="true" />
      ) : null}
      <Lines x={tx} y={top} lines={node.lines} className={styles.nodeLabel} lineHeight={LINE} anchor={Icon ? "start" : "middle"} />
      {sub.length ? <Lines x={cx} y={subTop} lines={sub} className={styles.sub} lineHeight={SMALL_LINE} /> : null}
    </>
  );
};

const Shape = ({ node }: { node: SceneNode }): React.ReactElement => {
  const { x, y, w, h } = node;
  const className = node.ghost ? styles.ghost : node.dashed ? `${styles.node} ${styles.dashed}` : styles.node;
  if (node.shape === "circle") return <circle className={className} cx={x + w / 2} cy={y + h / 2} r={w / 2} />;
  if (node.shape === "diamond") {
    return <polygon className={className} points={`${x + w / 2},${y} ${x + w},${y + h / 2} ${x + w / 2},${y + h} ${x},${y + h / 2}`} />;
  }
  if (node.shape === "store") {
    const ry = 5;
    return (
      <>
        <path className={className} d={`M${x},${y + ry}A${w / 2},${ry} 0 0 1 ${x + w},${y + ry}V${y + h - ry}A${w / 2},${ry} 0 0 1 ${x},${y + h - ry}Z`} />
        <path className={styles.storeRim} d={`M${x},${y + ry}A${w / 2},${ry} 0 0 0 ${x + w},${y + ry}`} />
      </>
    );
  }
  return <rect className={className} x={x} y={y} width={w} height={h} rx={node.shape === "pill" ? h / 2 : 0} />;
};

/** Where a path ends: the last two numbers of its last command. */
const endOf = (d: string) => {
  const n = d.match(/-?\d+(\.\d+)?/g)!.map(Number);
  return { x: n[n.length - 2], y: n[n.length - 1] };
};

/** Boxes, containers and lines from a frame's layout, as one static SVG. */
const FlowPlot: React.FC<FlowPlotProps> = ({ scene, id, variant, minScale }) => {
  const marker = `flow-arrow-${id}-${variant}`;
  return (
    <svg
      className={`${styles.plot} ${styles[variant]}`}
      viewBox={`0 0 ${scene.width} ${scene.height}`}
      style={{ maxWidth: scene.width, minWidth: Math.round(scene.width * minScale) }}
      aria-hidden="true"
    >
      <defs>
        <marker id={marker} viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
          <path className={styles.arrowHead} d="M0,0L8,4L0,8Z" />
        </marker>
      </defs>
      {scene.groups.map(g => (
        <rect key={g.id} className={styles.group} x={g.x} y={g.y} width={g.w} height={g.h} />
      ))}
      {scene.lifelines.map((l, i) => (
        <line key={i} className={styles.lifeline} x1={l.x} y1={l.y1} x2={l.x} y2={l.y2} />
      ))}
      {scene.edges.map((e, i) =>
        e.color ? (
          <g key={i} style={markColor(e.color)}>
            <path className={e.dashed ? `${styles.connector} ${styles.dashedLine}` : styles.connector} d={e.d} />
            <circle className={styles.connectorEnd} cx={endOf(e.d).x} cy={endOf(e.d).y} r={3} />
          </g>
        ) : (
          <path
            key={i}
            className={e.dashed ? `${styles.edge} ${styles.dashedLine}` : styles.edge}
            d={e.d}
            markerEnd={`url(#${marker})`}
            markerStart={e.both ? `url(#${marker})` : undefined}
          />
        )
      )}
      {scene.nodes.map(n => (
        <g key={n.id} style={markColor(n.color)}>
          <Shape node={n} />
          <NodeContent node={n} />
        </g>
      ))}
      {scene.groups.map(g => (
        <g key={g.id}>
          <text className={styles.groupLabel} x={g.x + 10} y={g.y + GROUP_HEAD / 2 + 3} dominantBaseline="central">
            {g.label}
          </text>
          {g.badge ? (
            <g style={markColor(g.badge.color ?? "teal")}>
              <rect className={styles.badge} x={g.x + g.w - g.badge.w - 6} y={g.y + 6} width={g.badge.w} height={GROUP_HEAD - 6} />
              <text className={styles.badgeLabel} x={g.x + g.w - g.badge.w / 2 - 6} y={g.y + 6 + (GROUP_HEAD - 6) / 2} textAnchor="middle" dominantBaseline="central">
                {g.badge.text}
              </text>
            </g>
          ) : null}
        </g>
      ))}
      {scene.edges.map((e, i) =>
        e.label ? (
          <Lines
            key={i}
            x={e.label.x}
            y={e.label.y - ((e.label.lines.length - 1) * SMALL_LINE) / 2}
            lines={e.label.lines}
            className={styles.edgeLabel}
            lineHeight={SMALL_LINE}
          />
        ) : null
      )}
    </svg>
  );
};

export default FlowPlot;
