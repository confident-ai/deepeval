import React from "react";
import styles from "./VoiceDiagram.module.scss";

type Tone = "bad" | "ok" | "skip" | "info";

interface Sound {
  start: number;
  end: number;
  /** How the sound ends: fades out, stops at full volume, or runs flat. */
  ending?: "fade" | "cut" | "flat";
  /** 0–1, how loud the sound is relative to the lane's height. */
  level?: number;
  /** Draw a near-silent line instead of a waveform (digital silence). */
  silent?: boolean;
}

interface Clip {
  start: number;
  end: number;
  label?: string;
}

interface Lane {
  label: string;
  role?: "caller" | "agent" | "neutral";
  sounds?: Sound[];
  /** Outlined boxes for the separate clips each turn carries. */
  clips?: Clip[];
}

interface Window {
  start: number;
  end: number;
  /** Lane indexes to cover; all lanes when left out. */
  lanes?: number[];
  label?: string;
  tone?: Tone;
}

interface Mark {
  at: number;
  lane: number;
  tone?: Tone;
  /** A word or two drawn above the point. */
  label: string;
}

interface VoiceDiagramProps {
  title?: string;
  /** Seconds shown on the time axis. */
  duration: number;
  lanes: Lane[];
  windows?: Window[];
  marks?: Mark[];
  /** Hide the seconds axis, for diagrams about shape rather than time. */
  hideAxis?: boolean;
  unit?: string;
  caption?: React.ReactNode;
}

const WIDTH = 600;
const LABEL_WIDTH = 58;
const RIGHT_PAD = 6;
const LANE_HEIGHT = 52;
// Space above a lane: room for point labels only where a lane has them.
const LABEL_ROOM = 22;
const PLAIN_GAP = 8;
const AXIS_HEIGHT = 22;
const BAR_STEP = 3;

// Deterministic "noise" so every render draws the same waveform.
const wobble = (x: number) => {
  const value = Math.sin(x * 12.9898 + 78.233) * 43758.5453;
  return 0.45 + 0.55 * Math.abs(value - Math.floor(value) - 0.5) * 2;
};

const VoiceDiagram: React.FC<VoiceDiagramProps> = ({
  title,
  duration,
  lanes,
  windows = [],
  marks = [],
  hideAxis = false,
  unit = "s",
  caption,
}) => {
  const plotWidth = WIDTH - LABEL_WIDTH - RIGHT_PAD;
  const x = (seconds: number) =>
    LABEL_WIDTH + (Math.max(0, Math.min(duration, seconds)) / duration) * plotWidth;
  const tops: number[] = [];
  let cursor = 0;
  lanes.forEach((_, index) => {
    const labelled =
      marks.some((mark) => mark.lane === index) ||
      windows.some(
        (window) =>
          window.label &&
          Math.min(...(window.lanes ?? lanes.map((_, i) => i))) === index,
      );
    cursor += labelled ? LABEL_ROOM : index === 0 ? 2 : PLAIN_GAP;
    tops.push(cursor);
    cursor += LANE_HEIGHT;
  });
  const laneTop = (index: number) => tops[index];
  const lanesBottom = cursor;
  const height = lanesBottom + (hideAxis ? 4 : AXIS_HEIGHT);

  const step = duration <= 2 ? 0.25 : duration <= 6 ? 1 : duration <= 16 ? 2 : 5;
  const ticks: number[] = [];
  for (let t = 0; t <= duration + 1e-9; t += step) ticks.push(+t.toFixed(2));

  return (
    <figure className={styles.figure}>
      {title && <figcaption className={styles.title}>{title}</figcaption>}
      <svg
        viewBox={`0 0 ${WIDTH} ${height}`}
        className={styles.svg}
        role="img"
        aria-label={title}
      >
        {lanes.map((lane, index) => {
          const top = laneTop(index);
          const labelled = (lane.clips ?? []).some((clip) => clip.label);
          // Leave room under the waveform for clip labels.
          const mid = top + LANE_HEIGHT / 2 - (labelled ? 5 : 0);
          const maxAmp = LANE_HEIGHT / 2 - (labelled ? 10 : 5);
          return (
            <g key={`lane-${index}`} className={styles[lane.role ?? "neutral"]}>
              <rect
                x={LABEL_WIDTH}
                y={top}
                width={plotWidth}
                height={LANE_HEIGHT}
               
                className={styles.track}
              />
              <text
                x={LABEL_WIDTH - 10}
                y={top + LANE_HEIGHT / 2}
                className={styles.laneLabel}
              >
                {lane.label}
              </text>
              <line
                x1={LABEL_WIDTH}
                x2={LABEL_WIDTH + plotWidth}
                y1={mid}
                y2={mid}
                className={styles.baseline}
              />
              {(lane.clips ?? []).map((clip, clipIndex) => (
                <g key={`clip-${clipIndex}`}>
                  <rect
                    x={x(clip.start)}
                    y={top + 3}
                    width={Math.max(2, x(clip.end) - x(clip.start))}
                    height={LANE_HEIGHT - 6}
                   
                    className={styles.clip}
                  />
                  {clip.label && (
                    <text
                      x={x(clip.start) + 5}
                      y={top + LANE_HEIGHT - 7}
                      className={styles.clipLabel}
                    >
                      {clip.label}
                    </text>
                  )}
                </g>
              ))}
              {(lane.sounds ?? []).map((sound, soundIndex) => {
                const from = x(sound.start);
                const to = x(sound.end);
                if (sound.silent) {
                  return (
                    <line
                      key={`sound-${soundIndex}`}
                      x1={from}
                      x2={to}
                      y1={mid}
                      y2={mid}
                      className={styles.silent}
                    />
                  );
                }
                const bars = [];
                const level = sound.level ?? 0.8;
                const ending = sound.ending ?? "fade";
                const span = Math.max(1, to - from);
                for (let bx = from + 1; bx < to - 0.5; bx += BAR_STEP) {
                  const progress = (bx - from) / span;
                  let envelope = 1;
                  if (progress < 0.08) envelope = 0.35 + (progress / 0.08) * 0.65;
                  if (ending === "fade" && progress > 0.7)
                    envelope = Math.max(0.06, 1 - (progress - 0.7) / 0.3);
                  const amp = maxAmp * level * envelope * wobble(bx);
                  bars.push(
                    <line
                      key={bx}
                      x1={bx}
                      x2={bx}
                      y1={mid - amp}
                      y2={mid + amp}
                      className={styles.bar}
                    />,
                  );
                }
                return <g key={`sound-${soundIndex}`}>{bars}</g>;
              })}
            </g>
          );
        })}

        {windows.map((window, index) => {
          const covered = window.lanes ?? lanes.map((_, i) => i);
          const top = laneTop(Math.min(...covered)) - 4;
          const bottom = laneTop(Math.max(...covered)) + LANE_HEIGHT + 4;
          const from = x(window.start);
          const width = Math.max(3, x(window.end) - from);
          return (
            <g key={`window-${index}`} className={styles[window.tone ?? "info"]}>
              <rect
                x={from}
                y={top}
                width={width}
                height={bottom - top}
               
                className={styles.window}
              />
              {window.label && (
                <text x={from + width / 2} y={top - 4} className={styles.windowLabel}>
                  {window.label}
                </text>
              )}
            </g>
          );
        })}

        {marks.map((mark, index) => {
          const cx = x(mark.at);
          const top = laneTop(mark.lane);
          const tone = mark.tone ?? "info";
          const anchor =
            cx > LABEL_WIDTH + plotWidth - 48
              ? "end"
              : cx < LABEL_WIDTH + 48
                ? "start"
                : "middle";
          return (
            <g key={`mark-${index}`} className={styles[tone]}>
              <line
                x1={cx}
                x2={cx}
                y1={top + 2}
                y2={top + LANE_HEIGHT - 2}
                className={styles.markLine}
              />
              <circle cx={cx} cy={top + 2} r={3} className={styles.markDot} />
              <text
                x={cx}
                y={top - 7}
                textAnchor={anchor}
                className={styles.markLabel}
              >
                {mark.label}
              </text>
            </g>
          );
        })}

        {!hideAxis && (
          <g>
            {ticks.map((tick) => (
              <g key={`tick-${tick}`}>
                <line
                  x1={x(tick)}
                  x2={x(tick)}
                  y1={lanesBottom + 4}
                  y2={lanesBottom + 8}
                  className={styles.tick}
                />
                <text x={x(tick)} y={lanesBottom + 19} className={styles.tickLabel}>
                  {tick}
                  {unit}
                </text>
              </g>
            ))}
          </g>
        )}
      </svg>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
    </figure>
  );
};

export default VoiceDiagram;
