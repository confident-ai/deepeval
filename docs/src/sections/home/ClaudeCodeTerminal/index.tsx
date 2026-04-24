"use client";

import Image from "next/image";
import styles from "./ClaudeCodeTerminal.module.scss";

/* Each row is a single on-screen line. Tool results are inlined with
 * their tool call so the whole session fits in 5 body lines. */

type Row =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | {
      kind: "tool";
      tool: "Bash" | "Edit";
      args: string;
      result: string;
      resultTone?: "neutral" | "warn" | "pass";
    };

const SCRIPT: Row[] = [
  {
    kind: "user",
    text: "eval the refund agent and fix any regressions",
  },
  {
    kind: "tool",
    tool: "Bash",
    args: "deepeval test run agents/checkout.py",
    result: "faithfulness 0.64 ⚠",
    resultTone: "warn",
  },
  {
    kind: "tool",
    tool: "Edit",
    args: "agents/retriever.py",
    result: "scoped to active refund policies",
  },
  {
    kind: "tool",
    tool: "Bash",
    args: "deepeval test run agents/checkout.py",
    result: "faithfulness 0.98 ✓",
    resultTone: "pass",
  },
  {
    kind: "assistant",
    text: "All metrics green — ready to commit.",
  },
];

const ClaudeCodeTerminal: React.FC = () => {
  return (
    <div
      className={styles.terminal}
      role="img"
      aria-label="Claude Code session that runs the agent, runs DeepEval, fixes a faithfulness regression, and re-runs until all metrics pass"
    >
      <div className={styles.chromeTop} aria-hidden>
        <span className={styles.chromeRuleStart} />
        <span className={styles.chromeLabel}>Claude Code v2.1.19</span>
        <span className={styles.chromeRuleEnd} />
      </div>

      <div className={styles.body}>
        <div className={styles.mascotWrap}>
          <Image
            src="/icons/claudecode.svg"
            alt="Claude Code"
            width={44}
            height={44}
            className={styles.mascot}
            priority={false}
          />
        </div>

        <div className={styles.lines}>
          {SCRIPT.map((row, i) => {
            const delay = {
              animationDelay: `${i * 0.1}s`,
            } as React.CSSProperties;

            if (row.kind === "user") {
              return (
                <div
                  key={i}
                  className={`${styles.line} ${styles.userLine}`}
                  style={delay}
                >
                  <span className={styles.userPrompt}>&gt;</span>
                  <span className={styles.userText}>{row.text}</span>
                </div>
              );
            }

            if (row.kind === "assistant") {
              return (
                <div
                  key={i}
                  className={`${styles.line} ${styles.assistantLine}`}
                  style={delay}
                >
                  <span className={styles.assistantDot}>●</span>
                  <span className={styles.assistantText}>{row.text}</span>
                </div>
              );
            }

            const resultToneClass =
              row.resultTone === "pass"
                ? styles.resultPass
                : row.resultTone === "warn"
                ? styles.resultWarn
                : styles.resultNeutral;

            return (
              <div
                key={i}
                className={`${styles.line} ${styles.toolLine}`}
                style={delay}
              >
                <span className={styles.toolBullet}>⏺</span>
                <span className={styles.toolName}>{row.tool}</span>
                <span className={styles.toolParen}>(</span>
                <span className={styles.toolArgs}>{row.args}</span>
                <span className={styles.toolParen}>)</span>
                <span className={styles.toolArrow}>⎿</span>
                <span className={`${styles.toolResult} ${resultToneClass}`}>
                  {row.result}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.inputBox}>
        <span className={styles.inputPrompt}>&gt;</span>
        <span className={styles.inputGhost}>Try &ldquo;ship it&rdquo;</span>
        <span className={styles.caret} aria-hidden />
      </div>
      <div className={styles.shortcuts}>? for shortcuts</div>
    </div>
  );
};


export default ClaudeCodeTerminal;
