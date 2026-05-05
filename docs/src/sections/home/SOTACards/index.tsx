"use client";

import type { ReactNode } from "react";
import styles from "./SOTACards.module.scss";

type Card = {
  icon: ReactNode;
  heading: string;
  description: string;
};

/* G-Eval glyph — chain-of-thought:
 * 4 nodes progressively "lighting up" left-to-right, connected by arrow segments.
 * Reads as: thought 1 → thought 2 → thought 3 → final score.
 */
const GEvalGlyph: React.FC = () => {
  const nodes = [10, 24, 38, 52];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {/* Connectors between nodes */}
      {nodes.slice(0, -1).map((x, i) => (
        <line
          key={`c-${i}`}
          x1={x + 4}
          y1="24"
          x2={nodes[i + 1] - 4}
          y2="24"
          className={styles.cotLink}
          style={{ animationDelay: `${i * 0.3 + 0.15}s` } as React.CSSProperties}
        />
      ))}
      {/* Nodes */}
      {nodes.map((x, i) => (
        <circle
          key={`n-${i}`}
          cx={x}
          cy="24"
          r="3"
          className={i === nodes.length - 1 ? styles.cotNodeFinal : styles.cotNode}
          style={{ animationDelay: `${i * 0.3}s` } as React.CSSProperties}
        />
      ))}
    </svg>
  );
};

/* DAG glyph — directed acyclic graph:
 * 5 nodes in a kite layout with 5 directed edges. Edges draw in sequence,
 * showing a flow from the entry node to a converged leaf.
 */
const DAGGlyph: React.FC = () => {
  /* Node coords (cx, cy):
   *       a (32,8)
   *      / \
   *   b(16,22)  c(48,22)
   *      \     /
   *       d(32,34)
   *         |
   *       e(32,44)   — final / output
   */
  const nodes = [
    { x: 32, y: 8 },
    { x: 16, y: 22 },
    { x: 48, y: 22 },
    { x: 32, y: 34 },
    { x: 32, y: 44 },
  ];

  /* Edges: (fromIndex, toIndex, animationDelay) */
  const edges: Array<[number, number, number]> = [
    [0, 1, 0],
    [0, 2, 0.15],
    [1, 3, 0.45],
    [2, 3, 0.45],
    [3, 4, 0.75],
  ];

  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {edges.map(([from, to, delay], i) => {
        const a = nodes[from];
        const b = nodes[to];
        return (
          <line
            key={`e-${i}`}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            className={styles.dagEdge}
            style={{ animationDelay: `${delay}s` } as React.CSSProperties}
          />
        );
      })}
      {nodes.map((n, i) => (
        <circle
          key={`n-${i}`}
          cx={n.x}
          cy={n.y}
          r={i === nodes.length - 1 ? 3 : 2.5}
          className={i === nodes.length - 1 ? styles.dagNodeFinal : styles.dagNode}
          style={{ animationDelay: `${i * 0.15}s` } as React.CSSProperties}
        />
      ))}
    </svg>
  );
};

/* QAG glyph — question → reference → answer:
 * Small Q block and A block with a document of text lines between them;
 * a pulse dot travels Q → doc → A to show reference-grounded generation.
 */
const QAGGlyph: React.FC = () => {
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {/* Path the pulse follows (invisible anchor) */}
      <path
        id="qag-path"
        d="M 10 24 L 32 24 L 54 24"
        fill="none"
        stroke="none"
      />

      {/* Q block */}
      <g className={styles.qagBlock}>
        <rect
          x="4"
          y="18"
          width="12"
          height="12"
          rx="2"
          className={styles.qagBlockRing}
        />
        <text
          x="10"
          y="27"
          className={styles.qagLabel}
          textAnchor="middle"
        >
          Q
        </text>
      </g>

      {/* Reference doc in the middle — lines representing source text */}
      <g className={styles.qagDoc}>
        <rect
          x="22"
          y="12"
          width="20"
          height="24"
          rx="1.5"
          className={styles.qagDocRing}
        />
        <line x1="25" y1="18" x2="39" y2="18" className={styles.qagDocLine} />
        <line x1="25" y1="22" x2="36" y2="22" className={styles.qagDocLine} />
        <line x1="25" y1="26" x2="38" y2="26" className={styles.qagDocLine} />
        <line x1="25" y1="30" x2="33" y2="30" className={styles.qagDocLine} />
      </g>

      {/* A block */}
      <g className={styles.qagBlock}>
        <rect
          x="48"
          y="18"
          width="12"
          height="12"
          rx="2"
          className={styles.qagBlockRing}
        />
        <text
          x="54"
          y="27"
          className={styles.qagLabel}
          textAnchor="middle"
        >
          A
        </text>
      </g>

      {/* Traveling pulse */}
      <circle r="2" className={styles.qagPulse} cy="24" />
    </svg>
  );
};

const CARDS: Card[] = [
  {
    icon: <GEvalGlyph />,
    heading: "G-Eval",
    description:
      "Criteria-based, chain-of-thought scoring via form-filling for reliable subjective evals.",
  },
  {
    icon: <DAGGlyph />,
    heading: "DAG",
    description:
      "Directed-acyclic-graph metrics for objective, multi-step conditional scoring.",
  },
  {
    icon: <QAGGlyph />,
    heading: "QAG",
    description:
      "Question-Answer Generation for close-ended, reference-grounded scoring.",
  },
];

const SOTACards: React.FC = () => {
  return (
    <div className={styles.grid}>
      {CARDS.map((card, i) => (
        <article key={i} className={styles.card}>
          <div className={styles.iconWrap}>{card.icon}</div>
          <h3 className={styles.heading}>{card.heading}</h3>
          <p className={styles.description}>{card.description}</p>
        </article>
      ))}
    </div>
  );
};


export default SOTACards;
