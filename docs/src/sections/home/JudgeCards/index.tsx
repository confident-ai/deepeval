"use client";

import type { CSSProperties, ReactNode } from "react";
import styles from "./JudgeCards.module.scss";

type Card = {
  icon: ReactNode;
  heading: string;
  description: string;
};

/* Animated glyph: multi-bar meter — many metrics, all live. */
const MetricsGlyph: React.FC = () => {
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      <line x1="6" x2="58" y1="40" y2="40" className={styles.glyphAxis} />
      {[
        { x: 10, delay: 0 },
        { x: 20, delay: 0.15 },
        { x: 30, delay: 0.3 },
        { x: 40, delay: 0.45 },
        { x: 50, delay: 0.6 },
      ].map((bar, i) => (
        <rect
          key={i}
          x={bar.x}
          y="14"
          width="6"
          height="26"
          rx="1.5"
          className={styles.glyphBar}
          style={{ animationDelay: `${bar.delay}s` }}
        />
      ))}
    </svg>
  );
};

/* Glyph: image frame — classic picture icon with a small sun and mountain silhouette. */
const MultiModalGlyph: React.FC = () => {
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {/* Frame */}
      <rect
        x="14"
        y="10"
        width="36"
        height="28"
        rx="2.5"
        className={styles.glyphImageFrame}
      />
      {/* Sun */}
      <circle cx="24" cy="18" r="2.6" className={styles.glyphImageSun} />
      {/* Mountains */}
      <path
        d="M14 34 L24 24 L30 29 L37 20 L50 32 L50 38 L14 38 Z"
        className={styles.glyphImageFill}
      />
    </svg>
  );
};

/* Animated glyph: conversation with per-turn scores — evals at every turn. */
const ConversationalEvalsGlyph: React.FC = () => {
  const rows = [
    { bubbleX: 6, bubbleW: 20, scoreCx: 32, side: "user", delay: 0 },
    { bubbleX: 24, bubbleW: 28, scoreCx: 58, side: "agent", delay: 0.3 },
    { bubbleX: 6, bubbleW: 16, scoreCx: 28, side: "user", delay: 0.6 },
  ];
  return (
    <svg
      viewBox="0 0 64 48"
      className={styles.glyph}
      aria-hidden
      focusable="false"
    >
      {rows.map((r, i) => (
        <g key={i}>
          <rect
            x={r.bubbleX}
            y={6 + i * 13}
            width={r.bubbleW}
            height="9"
            rx="2.5"
            className={`${styles.glyphConvBubble} ${
              r.side === "agent"
                ? styles.glyphConvBubbleAgent
                : styles.glyphConvBubbleUser
            }`}
          />
          <circle
            cx={r.scoreCx}
            cy={10.5 + i * 13}
            r="2.2"
            className={styles.glyphConvScore}
            style={{ animationDelay: `${r.delay}s` } as CSSProperties}
          />
        </g>
      ))}
    </svg>
  );
};

const CARDS: Card[] = [
  {
    icon: <MetricsGlyph />,
    heading: "50+ research-backed metrics",
    description:
      "Hallucination, faithfulness, answer relevancy, summarization, toxicity, bias, and more — ready out of the box.",
  },
  {
    icon: <ConversationalEvalsGlyph />,
    heading: "Native conversational evals",
    description:
      "Role adherence, knowledge retention, and conversation completeness — dedicated metrics built for multi-turn from day one.",
  },
  {
    icon: <MultiModalGlyph />,
    heading: "Multi-modal by default",
    description:
      "Text, images, and audio — all first-class. Same test case, same runner, same metrics across every modality.",
  },
];

const JudgeCards: React.FC = () => {
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


export default JudgeCards;
