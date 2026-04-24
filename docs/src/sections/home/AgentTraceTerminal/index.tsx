"use client";

import styles from "./AgentTraceTerminal.module.scss";

type LineKind =
  | "cmd"
  | "root"
  | "agent"
  | "tool"
  | "llm"
  | "retriever"
  | "blank"
  | "summary";

type TraceLine = {
  kind: LineKind;
  prefix?: string;
  name?: string;
  metric?: string;
  score?: string;
  duration?: string;
  pass?: boolean;
};

const TRACE: TraceLine[] = [
  { kind: "cmd", name: "deepeval test run agents/checkout.py" },
  { kind: "blank" },
  { kind: "root", prefix: "●", name: "test_checkout_agent" },
  { kind: "blank", prefix: "│" },
  {
    kind: "agent",
    prefix: "├─",
    name: "plan_refund_strategy",
    metric: "G-Eval",
    score: "0.94",
    duration: "220ms",
    pass: true,
  },
  {
    kind: "retriever",
    prefix: "│  ├─",
    name: "retrieve_policy_docs(query=…)",
    metric: "Context Recall",
    score: "0.89",
    duration: "68ms",
    pass: true,
  },
  {
    kind: "tool",
    prefix: "│  ├─",
    name: 'lookup_order(id="#9281")',
    metric: "Faithfulness",
    score: "1.00",
    duration: "45ms",
    pass: true,
  },
  {
    kind: "llm",
    prefix: "│  └─",
    name: "gpt-4o · classify_intent",
    metric: "Answer Relevancy",
    score: "0.92",
    duration: "130ms",
    pass: true,
  },
  { kind: "blank", prefix: "│" },
  {
    kind: "tool",
    prefix: "├─",
    name: "process_refund(amount=29.99)",
    metric: "deterministic",
    score: "✓",
    duration: "85ms",
    pass: true,
  },
  { kind: "blank", prefix: "│" },
  {
    kind: "llm",
    prefix: "└─",
    name: "gpt-4o · draft_response",
    metric: "Helpfulness",
    score: "0.88",
    duration: "195ms",
    pass: true,
  },
  { kind: "blank" },
  {
    kind: "summary",
    name: "Trace score  0.92   ·   5/5 metrics passed",
    pass: true,
  },
];

function kindLabel(kind: LineKind): string | null {
  switch (kind) {
    case "agent":
      return "AGENT";
    case "tool":
      return "TOOL";
    case "llm":
      return "LLM";
    case "retriever":
      return "RET";
    default:
      return null;
  }
}

const AgentTraceTerminal: React.FC = () => {
  return (
    <div className={styles.terminal} role="img" aria-label="Example agent trace with per-step metric scores">
      <div className={styles.bar}>
        <div className={styles.dots}>
          <span />
          <span />
          <span />
        </div>
        <span className={styles.title}>agent_trace · deepeval</span>
        <span className={styles.barSpacer} aria-hidden />
      </div>
      <div className={styles.body}>
        {TRACE.map((line, i) => (
          <div
            key={i}
            className={`${styles.line} ${styles[`line_${line.kind}`]}`}
            style={{ animationDelay: `${i * 0.11}s` } as React.CSSProperties}
          >
            {line.kind === "cmd" ? (
              <>
                <span className={styles.prompt}>$</span>
                <span className={styles.cmdText}>{line.name}</span>
              </>
            ) : line.kind === "summary" ? (
              <>
                <span className={styles.summaryDot} aria-hidden />
                <span className={styles.summaryText}>{line.name}</span>
                {line.pass && (
                  <span className={styles.summaryBadge}>passed</span>
                )}
              </>
            ) : line.kind === "blank" ? (
              <span className={styles.prefix}>{line.prefix ?? " "}</span>
            ) : line.kind === "root" ? (
              <>
                <span className={styles.rootDot}>{line.prefix}</span>
                <span className={styles.rootName}>{line.name}</span>
              </>
            ) : (
              <>
                <span className={styles.prefix}>{line.prefix}</span>
                <span
                  className={`${styles.badge} ${
                    styles[`badge_${line.kind}`]
                  }`}
                >
                  {kindLabel(line.kind)}
                </span>
                <span className={styles.name}>{line.name}</span>
                <span className={styles.meta}>
                  <span className={styles.metric}>{line.metric}</span>
                  <span
                    className={`${styles.score} ${
                      line.pass ? styles.scorePass : styles.scoreFail
                    }`}
                  >
                    {line.score}
                  </span>
                  <span className={styles.duration}>{line.duration}</span>
                </span>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};


export default AgentTraceTerminal;
