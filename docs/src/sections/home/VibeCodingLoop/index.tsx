import type { ReactNode } from "react";
import { Bot, Crosshair, Gauge, Sparkles } from "lucide-react";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import styles from "./VibeCodingLoop.module.scss";

/* --------------------------------------------------------------------
 * VibeCodingLoop
 *
 * A 4-node clockwise loop diagram showing how DeepEval closes the
 * vibe coding feedback loop:
 *
 *   Coding Agent  ─patches code─▶  Your AI App
 *        ▲                              │
 *   reads failures                  runs evals
 *        │                              ▼
 *   Scored Trace  ◀──scores spans──  DeepEval
 *
 * Center is anchored with the "Eval harness for vibe coding agents"
 * label. Arrows pulse around the loop continuously to convey motion.
 * ------------------------------------------------------------------ */

type NodeId = "tl" | "tr" | "br" | "bl";

type Node = {
  id: NodeId;
  icon: ReactNode;
  title: string;
  meta: string;
};

const NODES: Node[] = [
  {
    id: "tl",
    icon: <Bot size={14} aria-hidden />,
    title: "Coding Agent",
    meta: "Cursor · Claude Code · Codex",
  },
  {
    id: "tr",
    icon: <Sparkles size={14} aria-hidden />,
    title: "Your AI App",
    meta: "Agent · RAG · Chatbot",
  },
  {
    id: "br",
    icon: <Gauge size={14} aria-hidden />,
    title: "deepeval test run",
    meta: "50+ metrics, one CLI",
  },
  {
    id: "bl",
    icon: <Crosshair size={14} aria-hidden />,
    title: "Scored Trace",
    meta: "Span-level scores + reasons",
  },
];

type Arrow = {
  /* Quadratic bezier path for the connecting arc, drawn in
   * the SVG's 700×420 viewBox. Each arc bows inward toward the
   * center label so the four arcs together suggest a circular flow.*/
  d: string;
  /* End-point coordinates + tangent rotation for the arrowhead glyph. */
  arrow: { x: number; y: number; rotate: number };
  /* Caption rendered next to the arc midpoint. */
  label: string;
  labelX: number;
  labelY: number;
  /* Stagger: arrow N "fires" at delay N. Each fire takes ~1s; total cycle 4s. */
  delay: number;
};

const ARROWS: Arrow[] = [
  // 1. Top:    TL → TR (Coding Agent ──patches code──▶ Your AI App)
  {
    d: "M 260 95 Q 350 155 440 95",
    arrow: { x: 440, y: 95, rotate: -33.7 },
    label: "patches code",
    labelX: 350,
    labelY: 150,
    delay: 0,
  },
  // 2. Right:  TR → BR (Your AI App ──runs evals──▶ deepeval test run)
  {
    d: "M 560 160 Q 500 210 560 260",
    arrow: { x: 560, y: 260, rotate: 39.8 },
    label: "runs evals",
    labelX: 480,
    labelY: 213,
    delay: 1,
  },
  // 3. Bottom: BR → BL (deepeval ──scores spans──▶ Scored Trace)
  {
    d: "M 440 325 Q 350 265 260 325",
    arrow: { x: 260, y: 325, rotate: 146.3 },
    label: "scores spans",
    labelX: 350,
    labelY: 277,
    delay: 2,
  },
  // 4. Left:   BL → TL (Scored Trace ──reads failures──▶ Coding Agent)
  {
    d: "M 140 260 Q 200 210 140 160",
    arrow: { x: 140, y: 160, rotate: -140.2 },
    label: "reads failures",
    labelX: 220,
    labelY: 213,
    delay: 3,
  },
];

const VibeCodingLoop: React.FC = () => {
  return (
    <PauseOffscreen>
      <div
        className={styles.wrap}
        role="img"
        aria-label="The DeepEval vibe coding loop: coding agent patches code, your AI app runs deepeval test run, scored traces feed back to the coding agent."
      >
      {/* --- SVG layer: arcs, arrowheads, and arc labels --- */}
      <svg
        className={styles.svg}
        viewBox="0 0 700 420"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden
        focusable="false"
      >
        {/* Background arcs — always visible at low opacity. */}
        {ARROWS.map((a, i) => (
          <path key={`bg-${i}`} d={a.d} className={styles.arcBg} fill="none" />
        ))}

        {/* Foreground "flowing" arcs — each fills sequentially to suggest current. */}
        {ARROWS.map((a, i) => (
          <path
            key={`fg-${i}`}
            d={a.d}
            className={styles.arcFg}
            pathLength={100}
            fill="none"
            style={{ animationDelay: `${a.delay}s` }}
          />
        ))}

        {/* Arrowheads — pulse in sync with the flowing arc. */}
        {ARROWS.map((a, i) => (
          <g
            key={`ah-${i}`}
            transform={`translate(${a.arrow.x} ${a.arrow.y}) rotate(${a.arrow.rotate})`}
            className={styles.arrowhead}
            style={{ animationDelay: `${a.delay}s` }}
          >
            <path d="M 0 0 L -7 -3.5 L -7 3.5 Z" />
          </g>
        ))}

        {/* Arc labels — small mono-style captions next to each arrow. */}
        {ARROWS.map((a, i) => (
          <text
            key={`lb-${i}`}
            x={a.labelX}
            y={a.labelY}
            textAnchor="middle"
            className={styles.arcLabel}
          >
            {a.label}
          </text>
        ))}
      </svg>

      {/* --- Center anchor label --- */}
      <div className={styles.center} aria-hidden>
        <span className={styles.centerEyebrow}>DeepEval</span>
        <span className={styles.centerTitle}>
          Eval harness for
          <br />
          vibe coding agents
        </span>
      </div>

      {/* --- HTML cards (positioned to align with SVG endpoints) --- */}
      {NODES.map((node) => (
        <article
          key={node.id}
          className={`${styles.card} ${styles[`card_${node.id}`]}`}
        >
          <div className={styles.cardIcon}>{node.icon}</div>
          <h3 className={styles.cardTitle}>{node.title}</h3>
          <p className={styles.cardMeta}>{node.meta}</p>
        </article>
      ))}

      {/* --- Mobile fallback: vertical step list (SVG hidden on small screens) --- */}
      <ol className={styles.mobileList} aria-hidden>
        {NODES.map((node, i) => (
          <li key={node.id} className={styles.mobileItem}>
            <span className={styles.mobileStep}>{i + 1}</span>
            <div className={styles.mobileBody}>
              <span className={styles.mobileTitle}>{node.title}</span>
              <span className={styles.mobileMeta}>{node.meta}</span>
            </div>
          </li>
        ))}
        <li className={styles.mobileLoopBack} aria-hidden>
          ↑ back to coding agent · loop closes
        </li>
      </ol>
      </div>
    </PauseOffscreen>
  );
};

export default VibeCodingLoop;
