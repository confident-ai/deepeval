import styles from "./EnterprisePlatformMockup.module.scss";

type EnterprisePlatformMockupProps = {
  variant: "collaboration" | "tracing" | "deployment";
};

const statusItems = [
  { label: "Hallucination", value: "0.8%", tone: "good" },
  { label: "User sentiment", value: "92%", tone: "good" },
  { label: "Tool failures", value: "14", tone: "warn" },
];

const EnterprisePlatformMockup: React.FC<EnterprisePlatformMockupProps> = ({
  variant,
}) => {
  if (variant === "collaboration") {
    return (
      <figure className={styles.mockup} aria-label="No-code collaboration workflow mockup">
        <div className={styles.topbar}>
          <span>Confident AI</span>
          <span>Dataset review</span>
        </div>
        <div className={styles.collabGrid}>
          <div className={styles.sidebar}>
            {["PM review", "QA queue", "Domain expert", "Ready to automate"].map(
              (item, i) => (
                <span key={item} className={i === 1 ? styles.activeNav : ""}>
                  {item}
                </span>
              ),
            )}
          </div>
          <div className={styles.annotationCard}>
            <span className={styles.kicker}>Conversation #1842</span>
            <p className={styles.prompt}>
              User asked for refund policy. Agent gave the wrong exception.
            </p>
            <div className={styles.tags}>
              <span>incorrect policy</span>
              <span>needs escalation</span>
            </div>
            <div className={styles.comment}>
              <strong>QA note</strong>
              <span>Convert this failure into a reusable eval metric.</span>
            </div>
          </div>
          <div className={styles.reviewerStack}>
            {["Maya - PM", "Luis - QA", "Dr. Chen - SME"].map((name, i) => (
              <div key={name} className={styles.reviewer}>
                <span className={styles.avatar}>{name.slice(0, 1)}</span>
                <span>{name}</span>
                <span className={styles.check}>{i < 2 ? "approved" : "open"}</span>
              </div>
            ))}
          </div>
        </div>
      </figure>
    );
  }

  if (variant === "tracing") {
    return (
      <figure className={styles.mockup} aria-label="Production tracing dashboard mockup">
        <div className={styles.topbar}>
          <span>Production monitor</span>
          <span>Last 24h</span>
        </div>
        <div className={styles.dashboardGrid}>
          <div className={styles.chartPanel}>
            <div className={styles.chartHeader}>
              <span>Quality score</span>
              <strong>94.2</strong>
            </div>
            <div className={styles.chart}>
              {[42, 58, 51, 66, 62, 74, 70, 83, 78, 88, 84, 92].map(
                (height, i) => (
                  <span key={i} style={{ height: `${height}%` }} />
                ),
              )}
            </div>
          </div>
          <div className={styles.statusPanel}>
            {statusItems.map((item) => (
              <div key={item.label} className={styles.statusRow}>
                <span>{item.label}</span>
                <strong data-tone={item.tone}>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className={styles.tracePanel}>
            {["agent.run", "retrieve_policy", "call_refund_tool", "final_answer"].map(
              (span, i) => (
                <div key={span} className={styles.traceRow}>
                  <span className={styles.traceDot} />
                  <span>{span}</span>
                  <em>{[210, 84, 142, 390][i]}ms</em>
                </div>
              ),
            )}
          </div>
        </div>
      </figure>
    );
  }

  return (
    <figure className={styles.mockup} aria-label="Enterprise deployment admin mockup">
      <div className={styles.topbar}>
        <span>Organization admin</span>
        <span>12 workspaces</span>
      </div>
      <div className={styles.deployGrid}>
        <div className={styles.orgTree}>
          {["Consumer AI", "Support Agents", "Risk & Compliance", "Internal Tools"].map(
            (team, i) => (
              <div key={team} className={styles.orgRow}>
                <span className={styles.orgIcon}>{i + 1}</span>
                <span>{team}</span>
                <em>{["18", "42", "9", "23"][i]} users</em>
              </div>
            ),
          )}
        </div>
        <div className={styles.controlsPanel}>
          <span className={styles.kicker}>Org controls</span>
          {["SSO enforced", "Audit logs on", "EU data region", "Custom retention"].map(
            (control) => (
              <div key={control} className={styles.controlRow}>
                <span>{control}</span>
                <strong>on</strong>
              </div>
            ),
          )}
        </div>
        <div className={styles.deployCard}>
          <strong>Self-hosted cluster</strong>
          <span>Updated 10 minutes ago</span>
          <div className={styles.progressTrack}>
            <span />
          </div>
        </div>
      </div>
    </figure>
  );
};

export default EnterprisePlatformMockup;
