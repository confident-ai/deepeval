"use client";

import { type ReactNode, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { FileCode2, LoaderCircle, Play, TerminalSquare } from "lucide-react";
import styles from "./HomePytestDemo.module.scss";

const codeLines = [
  <>
    <span className={styles.codeKeyword}>from</span>{" "}
    <span className={styles.codeModule}>deepeval.tracing</span>{" "}
    <span className={styles.codeKeyword}>import</span>{" "}
    <span className={styles.codeFunction}>observe</span>
  </>,
  <></>,
  <>
    <span className={styles.codeDecorator}>@observe()</span>
  </>,
  <>
    <span className={styles.codeKeyword}>def</span>{" "}
    <span className={styles.codeFunction}>test_checkout_agent</span>
    <span className={styles.codePunctuation}>():</span>
  </>,
  <>
    <span className={styles.codeIndent}>    </span>
    <span className={styles.codeVariable}>result</span>{" "}
    <span className={styles.codeOperator}>=</span>{" "}
    <span className={styles.codeFunction}>checkout_agent</span>
    <span className={styles.codePunctuation}>(</span>
    <span className={styles.codeString}>"refund duplicate charge"</span>
    <span className={styles.codePunctuation}>)</span>
  </>,
  <>
    <span className={styles.codeIndent}>    </span>
    <span className={styles.codeKeyword}>assert</span>{" "}
    <span className={styles.codeVariable}>result.status</span>{" "}
    <span className={styles.codeOperator}>==</span>{" "}
    <span className={styles.codeString}>"resolved"</span>
  </>,
  <>
    <span className={styles.codeIndent}>    </span>
    <span className={styles.codeKeyword}>assert</span>{" "}
    <span className={styles.codeVariable}>result.metrics</span>
    <span className={styles.codePunctuation}>[</span>
    <span className={styles.codeString}>"task_completion"</span>
    <span className={styles.codePunctuation}>]</span>{" "}
    <span className={styles.codeOperator}>&gt;=</span>{" "}
    <span className={styles.codeNumber}>0.92</span>
  </>,
];

const command = "deepeval test run tests/test_checkout_agent.py";

const timeline = [
  { delayMs: 350, line: command, tone: "command" as const },
  { delayMs: 950, line: "Calling checkout_agent() with traced inputs...", tone: "deepeval" as const },
  { delayMs: 2150, line: "Agent response captured. Starting evaluation suite...", tone: "muted" as const },
  { delayMs: 5200, line: "SUMMARY_LINE", tone: "summary" as const },
  { delayMs: 5600, line: "", tone: "muted" as const },
  { delayMs: 5900, line: "TABLE_START", tone: "table" as const },
  { delayMs: 6200, line: "goldens: 38 · traces: 38 · tools: 4 · p95 latency: 2.1s", tone: "muted" as const },
];

type HomePytestDemoProps = {
  hideHeader?: boolean;
};

type DemoBlockLanguage = "bash" | "python";

type ColabTerminalBlockProps = {
  content: ReactNode;
  language: DemoBlockLanguage;
  hideHeader?: boolean;
  browserButtons?: boolean;
  headerLogo?: ReactNode;
  title?: string;
  headerRight?: ReactNode;
  bodyClassName?: string;
  rootClassName?: string;
};

const ColabTerminalBlock: React.FC<ColabTerminalBlockProps> = ({
  content,
  language,
  hideHeader = false,
  browserButtons = true,
  headerLogo,
  title,
  headerRight,
  bodyClassName,
  rootClassName,
}) => {
  const rootClass = rootClassName
    ? `${styles.fusedBlock} ${rootClassName}`
    : styles.fusedBlock;
  const contentClass = bodyClassName
    ? `${styles.blockBody} ${bodyClassName}`
    : styles.blockBody;
  const effectiveLogo =
    headerLogo ??
    (language === "python" ? <FileCode2 size={13} /> : <TerminalSquare size={13} />);

  return (
    <div className={rootClass}>
      {!hideHeader ? (
        <div className={styles.blockHeader}>
          <div className={styles.blockHeaderLeft}>
            {browserButtons ? (
              <span className={styles.windowDots} aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
            ) : null}
            <span className={styles.headerLogo}>{effectiveLogo}</span>
            {title ? <span className={styles.panelLabel}>{title}</span> : null}
          </div>
          <div className={styles.blockHeaderRight}>
            {headerRight ? <span>{headerRight}</span> : null}
          </div>
        </div>
      ) : null}
      <div className={contentClass}>{content}</div>
    </div>
  );
};

const HomePytestDemo: React.FC<HomePytestDemoProps> = ({ hideHeader = false }) => {
  const [status, setStatus] = useState<"idle" | "running" | "done">("idle");
  const [visibleLineCount, setVisibleLineCount] = useState(0);

  useEffect(() => {
    if (status !== "running") return;

    const timers = timeline.map((step, index) =>
      window.setTimeout(() => {
        setVisibleLineCount(index + 1);
      }, step.delayMs),
    );

    const finishTimer = window.setTimeout(() => {
      setStatus("done");
    }, timeline[timeline.length - 1].delayMs + 350);

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
      window.clearTimeout(finishTimer);
    };
  }, [status]);

  const terminalLines = useMemo(
    () => timeline.slice(0, visibleLineCount),
    [visibleLineCount],
  );

  const appProgress =
    status === "idle" ? 0 : Math.min(100, Math.round((visibleLineCount / 2) * 100));
  const metricsProgress =
    status === "idle"
      ? 0
      : Math.min(100, Math.max(0, Math.round(((visibleLineCount - 1) / 3) * 100)));

  function runDemo() {
    setVisibleLineCount(0);
    setStatus("running");
  }
  return (
    <section className={styles.demo}>
      <ColabTerminalBlock
        language="python"
        hideHeader={hideHeader}
        browserButtons
        headerLogo={
          <Image
            src="/icons/python.svg"
            alt="Python"
            width={13}
            height={13}
            className={styles.headerLogoImage}
          />
        }
        title="tests/test_checkout_agent.py"
        bodyClassName={styles.codeBlock}
        rootClassName={styles.codePanel}
        content={
          <pre>
            {codeLines.map((line, index) => (
              <div key={index} className={styles.codeLine}>
                {line}
              </div>
            ))}
          </pre>
        }
      />

      <div className={styles.runtimePanel}>
        <ColabTerminalBlock
          language="bash"
          hideHeader={hideHeader}
          browserButtons={true}
          headerLogo={<TerminalSquare size={13} />}
          title="scripts/run_deepeval.sh"
          bodyClassName={styles.terminalBody}
          rootClassName={styles.terminal}
          content={
            <>
              {status === "idle" ? (
                <div className={`${styles.terminalLine} ${styles.commandLine}`}>
                  <span className={styles.prompt}>$</span>
                  <span>{command}</span>
                </div>
              ) : null}

              {terminalLines.map((step) => {
                if (step.tone === "table") {
                  return (
                    <div key={`${step.delayMs}-${step.line}`} className={styles.tableWrap}>
                      <div className={`${styles.tableRow} ${styles.tableTitleRow}`}>
                        <span className={styles.tableTitle}>Test Run Summary</span>
                      </div>
                      <div className={styles.tableRow}>
                        <span className={styles.tableCellHead}>metric</span>
                        <span className={styles.tableCellHead}>avg score</span>
                        <span className={styles.tableCellHead}>pass</span>
                        <span className={styles.tableCellHead}>fail</span>
                        <span className={styles.tableCellHead}>skip</span>
                        <span className={styles.tableCellHead}>p95</span>
                        <span className={styles.tableCellHead}>notes</span>
                      </div>
                      <div className={styles.tableRow}>
                        <span className={styles.tableCell}>Task completion</span>
                        <span className={`${styles.tableCell} ${styles.tableScore}`}>0.94</span>
                        <span className={`${styles.tableCell} ${styles.tablePass}`}>34</span>
                        <span className={`${styles.tableCell} ${styles.tableFail}`}>2</span>
                        <span className={`${styles.tableCell} ${styles.tableSkip}`}>2</span>
                        <span className={styles.tableCell}>1.8s</span>
                        <span className={styles.tableCell}>2 unresolved refund flows</span>
                      </div>
                      <div className={styles.tableRow}>
                        <span className={styles.tableCell}>Tool correctness</span>
                        <span className={`${styles.tableCell} ${styles.tableWarn}`}>0.72  ⚠️</span>
                        <span className={`${styles.tableCell} ${styles.tablePass}`}>27</span>
                        <span className={`${styles.tableCell} ${styles.tableFail}`}>9</span>
                        <span className={`${styles.tableCell} ${styles.tableSkip}`}>2</span>
                        <span className={styles.tableCell}>1.1s</span>
                        <span className={styles.tableCell}>refund.lookup arg mismatch</span>
                      </div>
                      <div className={styles.tableRow}>
                        <span className={styles.tableCell}>Faithfulness</span>
                        <span className={`${styles.tableCell} ${styles.tableWarn}`}>0.64  ⚠️</span>
                        <span className={`${styles.tableCell} ${styles.tablePass}`}>24</span>
                        <span className={`${styles.tableCell} ${styles.tableFail}`}>11</span>
                        <span className={`${styles.tableCell} ${styles.tableSkip}`}>3</span>
                        <span className={styles.tableCell}>1.6s</span>
                        <span className={styles.tableCell}>unsupported refund claims</span>
                      </div>
                      <div className={`${styles.tableRow} ${styles.tableRowSummary}`}>
                        <span className={styles.tableCell}>Overall</span>
                        <span className={`${styles.tableCell} ${styles.result}`}>0.77</span>
                        <span className={`${styles.tableCell} ${styles.tablePass}`}>34</span>
                        <span className={`${styles.tableCell} ${styles.tableFail}`}>3</span>
                        <span className={`${styles.tableCell} ${styles.tableSkip}`}>1</span>
                        <span className={styles.tableCell}>2.1s</span>
                        <span className={styles.tableCell}>tooling + grounding need work</span>
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={`${step.delayMs}-${step.line}`}
                    className={`${styles.terminalLine} ${styles[step.tone]} ${
                      step.line.startsWith("goldens:") ? styles.suiteMeta : ""
                    }`}
                  >
                    {step.tone === "command" ? (
                      <>
                        <span className={styles.prompt}>$</span>
                        <span>{step.line}</span>
                      </>
                    ) : step.tone === "summary" ? (
                      <>
                        <span className={styles.tablePass}>34 passed</span>
                        <span className={styles.summarySeparator}>, </span>
                        <span className={styles.tableFail}>3 failed</span>
                        <span className={styles.summarySeparator}>, </span>
                        <span className={styles.tableSkip}>1 skipped</span>
                        <span className={styles.summarySeparator}> in 6.84s</span>
                      </>
                    ) : (
                      step.line
                    )}
                  </div>
                );
              })}

              {status === "running" ? (
                <div className={styles.progressGroup}>
                  <div className={`${styles.terminalLine} ${styles.progressIntro}`}>
                    <span className={styles.inlineDots} aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                    <span>Running evals</span>
                  </div>

                  <div className={styles.progressLine}>
                    <span className={styles.progressLabel}>app.run()</span>
                    <span className={styles.progressTrack}>
                      <span
                        className={styles.progressFill}
                        style={{ width: `${appProgress}%` }}
                      />
                    </span>
                    <span className={styles.progressPct}>{appProgress}%</span>
                  </div>

                  <div className={styles.progressLine}>
                    <span className={styles.progressLabel}>metrics.eval()</span>
                    <span className={styles.progressTrack}>
                      <span
                        className={styles.progressFillAlt}
                        style={{ width: `${metricsProgress}%` }}
                      />
                    </span>
                    <span className={styles.progressPct}>{metricsProgress}%</span>
                  </div>
                </div>
              ) : null}

              {status === "running" ? <div className={styles.cursor} /> : null}
            </>
          }
        />

        <button
          type="button"
          className={styles.runButton}
          onClick={runDemo}
          disabled={status === "running"}
          data-button
          data-callout
        >
          {status === "running" ? (
            <LoaderCircle size={14} className={styles.spinner} aria-hidden="true" />
          ) : (
            <Play size={14} aria-hidden="true" />
          )}
          {status === "running" ? "Evaluating" : "Evaluate"}
        </button>
      </div>
    </section>
  );
};


export default HomePytestDemo;
