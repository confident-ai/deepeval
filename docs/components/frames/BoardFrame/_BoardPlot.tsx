import type * as React from "react";
import { type Box, textWidth } from "@/components/frames/FlowPlot/_scene";
import { markColor } from "@/components/frames/FrameShell/FrameShell";
import type { ChartColor } from "@/lib/charts/schema";
import {
  ARROW_LENGTH,
  type BoardScene,
  END_CIRCLE,
  END_DOT,
  MAGNIFIER,
  MIN_SCALE,
  type SceneEnd,
  type ScenePulloutLine,
  type SceneMark,
  type SceneText,
} from "./_scene";
import { BoardIcon } from "./_BoardIcon";
import { ELLIPSIS, TIER, sizeVars } from "./_tokens";
import styles from "./styles.module.scss";

interface BoardPlotProps {
  scene: BoardScene;
  id: string;
}

const rectOf = (b: Box, rx: number) => ({ x: b.x, y: b.y, width: b.w, height: b.h, rx });

interface TextProps {
  text: SceneText;
  tone?: "muted";
  halo?: boolean;
}

const CODE_SPAN = /(`[^`]+`)/;
const isCode = (part: string) => part.length > 2 && part.startsWith("`") && part.endsWith("`");
const CHIP_PAD = 4;

// `backticks` draw as inline code; a line that is only code also gets a chip behind it.
const Text = ({ text, tone, halo }: TextProps): React.ReactElement => {
  const parts = text.text.split(CODE_SPAN).filter(Boolean);
  const size = TIER[text.tier].text;
  const chip = parts.length === 1 && isCode(parts[0]) ? textWidth(parts[0].slice(1, -1), size) + 2 * CHIP_PAD : 0;
  const chipX = text.anchor === "middle" ? text.x - chip / 2 : text.anchor === "end" ? text.x - chip + CHIP_PAD : text.x - CHIP_PAD;
  return (
    <>
      {chip ? <rect className={styles.codeChip} x={chipX} y={text.y - size / 2 - 3} width={chip} height={size + 6} rx={3} /> : null}
      <text
        className={styles.text}
        data-tier={text.tier}
        data-tone={tone}
        data-halo={halo || undefined}
        data-weight={text.weight}
        x={text.x}
        y={text.y}
        textAnchor={text.anchor}
        dominantBaseline="central"
      >
        {parts.map((part, i) =>
          isCode(part) ? (
            <tspan key={i} className={styles.code}>
              {part.slice(1, -1)}
            </tspan>
          ) : (
            part
          )
        )}
      </text>
    </>
  );
};

/** A group drawn in a chart color, or in the board's own colors without one. */
const tint = (color: ChartColor | undefined, className?: string) => ({
  className: [className, color ? styles.tinted : undefined].filter(Boolean).join(" ") || undefined,
  style: color ? markColor(color) : undefined,
});

const Mark = ({ mark, rx, hatch }: { mark: SceneMark; rx: number; hatch?: string }): React.ReactElement => (
  <g {...tint(mark.color)}>
    <rect className={mark.color ? styles.mark : styles.note} {...rectOf(mark, rx)} />
    {hatch ? <rect className={styles.hatchFill} {...rectOf(mark, rx)} fill={hatch} /> : null}
    <Text text={mark.text} />
  </g>
);

const End = ({ end }: { end: SceneEnd }): React.ReactElement | null => {
  const { at, dir } = end;
  if (end.shape === "magnifier") {
    return (
      <>
        <circle className={styles.endHollow} cx={at.x} cy={at.y} r={MAGNIFIER} />
        <path className={styles.endHollow} d={`M${at.x - MAGNIFIER},${at.y}h${2 * MAGNIFIER}M${at.x},${at.y - MAGNIFIER}v${2 * MAGNIFIER}`} />
      </>
    );
  }
  if (end.shape === "circle") return <circle className={styles.endHollow} cx={at.x} cy={at.y} r={END_CIRCLE} />;
  if (end.shape === "dot") return <circle className={styles.endSolid} cx={at.x} cy={at.y} r={END_DOT} />;
  if (end.shape === "arrow") {
    const base = { x: at.x - dir.x * ARROW_LENGTH, y: at.y - dir.y * ARROW_LENGTH };
    const half = { x: -dir.y * (ARROW_LENGTH / 2), y: dir.x * (ARROW_LENGTH / 2) };
    return <polygon className={styles.endSolid} points={`${at.x},${at.y} ${base.x + half.x},${base.y + half.y} ${base.x - half.x},${base.y - half.y}`} />;
  }
  return null;
};

const PulloutLine = ({ line, children }: { line: ScenePulloutLine; children: React.ReactNode }): React.ReactElement => (
  <g {...tint(line.color, styles.pulloutLine)} data-colored={line.color ? "" : undefined}>
    {children}
  </g>
);

const BoardPlot: React.FC<BoardPlotProps> = ({ scene, id }) => {
  const arrow = `board-arrow-${id}`;
  const tip = `board-tip-${id}`;
  const hatchId = (color: ChartColor) => `board-hatch-${id}-${color}`;
  const hatched = new Set(scene.hatch);
  const { view, radius } = scene;

  const hatchOf = (mark: { color?: ChartColor }) => (mark.color && hatched.has(mark.color) ? `url(#${hatchId(mark.color)})` : undefined);

  return (
    <svg
      className={styles.plot}
      viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
      style={{ ...sizeVars, maxWidth: view.w, minWidth: Math.round(view.w * MIN_SCALE) }}
      aria-hidden="true"
    >
      <defs>
        <marker id={arrow} viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
          <path className={styles.arrowHead} d="M0,0L8,4L0,8Z" />
        </marker>
        <marker id={tip} viewBox="0 0 8 8" refX="8" refY="4" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto">
          <path className={styles.arrowHead} d="M0,0L8,4L0,8Z" />
        </marker>
        {scene.hatch.map(color => (
          <pattern
            key={color}
            id={hatchId(color)}
            width="6"
            height="6"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
            {...tint(color)}
          >
            <line className={styles.hatch} x1="0" y1="0" x2="0" y2="6" />
          </pattern>
        ))}
      </defs>
      <rect className={styles.panel} data-border={scene.panel.border} {...rectOf(scene.panel, radius.panel)} />
      {scene.heading ? <Text text={scene.heading} /> : null}
      {scene.mainBox ? <rect className={styles.tree} data-border={scene.mainBox.border} {...rectOf(scene.mainBox, radius.box)} /> : null}
      {scene.pullouts.flatMap((pullout, i) =>
        pullout.lines.map((line, j) => (
          <PulloutLine key={`${i}-${j}`} line={line}>
            <polyline className={styles.leader} data-style={line.style} points={line.points.map(p => `${p.x},${p.y}`).join(" ")} />
          </PulloutLine>
        ))
      )}
      {scene.main.boxes.map(box => (
        <g key={box.id} {...tint(box.color, styles.box)}>
          {box.shape === "ellipse" ? (
            <>
              <ellipse className={styles.boxShape} cx={box.x + box.w / 2} cy={box.y + box.h / 2} rx={box.w / 2} ry={box.h / 2} />
              {box.color && hatchOf(box) ? <ellipse className={styles.hatchFill} cx={box.x + box.w / 2} cy={box.y + box.h / 2} rx={box.w / 2} ry={box.h / 2} fill={hatchOf(box)} /> : null}
            </>
          ) : box.shape === "circle" ? (
            <>
              <circle className={box.wash ? styles.setShape : styles.boxShape} data-border={box.border} cx={box.x + box.w / 2} cy={box.y + box.h / 2} r={box.w / 2} />
              {box.color && !box.wash && hatchOf(box) ? <circle className={styles.hatchFill} cx={box.x + box.w / 2} cy={box.y + box.h / 2} r={box.w / 2} fill={hatchOf(box)} /> : null}
            </>
          ) : (
            <>
              <rect className={styles.boxShape} data-border={box.border} {...rectOf(box, box.holds ? radius.box : radius.card)} />
              {box.color && hatchOf(box) ? <rect className={styles.hatchFill} {...rectOf(box, box.holds ? radius.box : radius.card)} fill={hatchOf(box)} /> : null}
            </>
          )}
        </g>
      ))}
      {scene.main.rules.map((rule, i) => (
        <g key={i} {...tint(rule.color, styles.box)}>
          <path className={styles.rule} data-border={rule.border} d={rule.d} />
        </g>
      ))}
      {scene.main.labels.map((label, i) => (
        <Text key={i} text={label} />
      ))}
      {scene.main.dots.map((dots, i) =>
        [-1, 0, 1].map(k => (
          <circle
            key={`${i}-${k}`}
            className={styles.ellipsis}
            cx={dots.x + (dots.vertical ? 0 : k * ELLIPSIS.pitch)}
            cy={dots.y + (dots.vertical ? k * ELLIPSIS.pitch : 0)}
            r={ELLIPSIS.r}
          />
        ))
      )}
      {scene.main.elbows.map((d, i) => (
        <path key={i} className={styles.line} d={d} markerEnd={`url(#${tip})`} />
      ))}
      {scene.main.links.map((link, i) => (
        <path key={i} className={styles.line} data-dashed={link.dashed || undefined} d={link.d} markerStart={link.start ? `url(#${arrow})` : undefined} markerEnd={link.end ? `url(#${arrow})` : undefined} />
      ))}
      {scene.arrows.map((a, i) => (
        <path key={i} className={styles.line} d={a.d} markerStart={a.start ? `url(#${arrow})` : undefined} markerEnd={a.end ? `url(#${arrow})` : undefined} />
      ))}
      {scene.main.bars.map(bar => (
        <Mark key={bar.id} mark={bar} rx={radius.mark} hatch={hatchOf(bar)} />
      ))}
      {scene.main.icons.map((icon, i) => (
        <BoardIcon key={i} icon={icon} />
      ))}
      {scene.columnHeadings.map((text, i) => (
        <Text key={i} text={text} halo />
      ))}
      {scene.notes.map((note, i) =>
        note.bare ? (
          <Text key={i} text={note.text} />
        ) : (
          <Mark key={i} mark={note} rx={radius.mark} hatch={hatchOf(note)} />
        )
      )}
      {[...scene.arrowLabels, ...scene.main.linkLabels].map((text, i) => (
        <Text key={i} text={text} tone="muted" halo />
      ))}
      {scene.pullouts.map((pullout, i) => (
        <g key={i}>
          {pullout.shape === "circle" ? (
            <>
              <circle className={styles.lens} cx={pullout.circle.x} cy={pullout.circle.y} r={pullout.circle.r} />
              <rect className={styles.card} {...rectOf(pullout.card, radius.card)} />
            </>
          ) : null}
          {pullout.heading ? <Text text={pullout.heading} /> : null}
          {pullout.pills.map((pill, j) => (
            <Mark key={j} mark={pill} rx={radius.mark} hatch={hatchOf(pill)} />
          ))}
        </g>
      ))}
      {scene.icons.map((icon, i) => (
        <BoardIcon key={i} icon={icon} />
      ))}
      {scene.pullouts.flatMap((pullout, i) =>
        pullout.lines.map((line, j) => (
          <PulloutLine key={`${i}-${j}`} line={line}>
            <End end={line.start} />
            <End end={line.end} />
            {line.label ? <Text text={line.label} tone="muted" halo /> : null}
          </PulloutLine>
        ))
      )}
    </svg>
  );
};

export default BoardPlot;
