"use client";

import { useEffect, useState, type CSSProperties } from "react";
import styles from "./DatasetDemos.module.scss";

type Golden = {
  id: string;
  question: string;
  answer: string;
  tag: "standard" | "variation" | "edge case" | "adversarial";
};

const GOLDENS: Golden[] = [
  {
    id: "g_01",
    question: "How do I refund an order?",
    answer: "Call POST /refunds with order_id and amount.",
    tag: "standard",
  },
  {
    id: "g_02",
    question: "Can I partially refund a line item?",
    answer: "Yes — include line_item_ids in the POST /refunds body.",
    tag: "variation",
  },
  {
    id: "g_03",
    question:
      "If the order already shipped, can I still refund without returning it?",
    answer: "Shipped orders follow the return flow — call POST /returns first.",
    tag: "edge case",
  },
  {
    id: "g_04",
    question: "Refund WITHOUT order_id pls!!!!",
    answer: "order_id is required. Politely ask the user to share it.",
    tag: "adversarial",
  },
];

const SOURCE_DOCS = [
  {
    name: "docs/billing-api.md",
    lines: [88, 72, 92, 58, 78],
  },
  {
    name: "schemas/refund.json",
    lines: [62, 85, 48, 70],
  },
  {
    name: "contracts/orders.yaml",
    lines: [75, 65, 90, 55, 80, 45],
  },
];

const STAGES = [
  "Chunking",
  "Extracting context",
  "Generating",
  "Evolving",
  "Filtering",
  "Applying styles",
  "Done",
] as const;

const STAGE_INTERVAL_MS = 1000; // each stage holds for 1s → full pipeline runs once in 7s

export const GoldenGenerationDemo: React.FC = () => {
  // activeStage advances one step past the last index (STAGES.length) so that
  // the final "done" entry also flips from its active state to a ticked/
  // completed state once the pipeline fully settles.
  const [activeStage, setActiveStage] = useState(0);
  const isDone = activeStage >= STAGES.length;

  useEffect(() => {
    if (isDone) return;
    const id = setInterval(() => {
      setActiveStage((s) => {
        if (s >= STAGES.length) {
          clearInterval(id);
          return s;
        }
        return s + 1;
      });
    }, STAGE_INTERVAL_MS);
    return () => clearInterval(id);
  }, [isDone]);

  return (
    <div className={styles.panel} data-done={isDone ? "true" : undefined}>
      <ol className={styles.stages} aria-label="generation pipeline">
        {STAGES.map((stage, i) => {
          const state =
            i < activeStage ? "done" : i === activeStage ? "active" : "pending";
          return (
            <li
              key={stage}
              className={`${styles.stage} ${
                styles[`stage_${state}` as keyof typeof styles]
              }`}
              aria-current={state === "active" ? "step" : undefined}
            >
              <span className={styles.stageMark} aria-hidden>
                {state === "done" ? "✓" : state === "active" ? "●" : "○"}
              </span>
              <span className={styles.stageLabel}>{stage}</span>
            </li>
          );
        })}
      </ol>

      <div className={styles.goldenLayout}>
        {/* LEFT: source docs stack */}
        <aside className={styles.source}>
          <div className={styles.sourceLabel}>SOURCES</div>
          {SOURCE_DOCS.map((doc, i) => (
            <div key={doc.name} className={styles.sourceDoc}>
              <div className={styles.sourceDocName}>{doc.name}</div>
              {doc.lines.map((w, j) => (
                <div
                  key={j}
                  className={styles.sourceLine}
                  style={{ width: `${w}%` } as CSSProperties}
                />
              ))}
              <div
                className={styles.sourceScan}
                aria-hidden
                style={{ animationDelay: `${i * 1.1}s` } as CSSProperties}
              />
            </div>
          ))}
        </aside>

        {/* MIDDLE: particle flow */}
        <div className={styles.flow} aria-hidden>
          {[0, 0.9, 1.8].map((delay, i) => (
            <span
              key={i}
              className={styles.flowPulse}
              style={{ animationDelay: `${delay}s` } as CSSProperties}
            />
          ))}
        </div>

        {/* RIGHT: generated goldens stacking up */}
        <div className={styles.goldens}>
          <div className={styles.goldenLabel}>GOLDENS</div>
          {GOLDENS.map((g, i) => (
            <article
              key={g.id}
              className={styles.goldenCard}
              style={
                {
                  animationDelay: `${0.45 + i * 0.55}s`,
                } as CSSProperties
              }
            >
              <header className={styles.goldenHead}>
                <span className={styles.goldenId}>{g.id}</span>
                <span
                  className={`${styles.goldenTag} ${
                    styles[
                      `tag_${g.tag.replace(/\s+/g, "_")}` as keyof typeof styles
                    ]
                  }`}
                >
                  {g.tag}
                </span>
              </header>
              <p className={styles.goldenQ}>
                <span className={styles.goldenQALabel}>Q</span>
                {g.question}
              </p>
              <p className={styles.goldenA}>
                <span className={styles.goldenQALabel}>A</span>
                {g.answer}
              </p>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
};
