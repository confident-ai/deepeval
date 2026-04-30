import styles from "./TraceLoopConnector.module.scss";

/* Visual connector that sits between the AgentTraceTerminal above
 * and the ClaudeCodeTerminal below to make the loop explicit:
 *
 *    trace → (down, "evaluate") → Claude reads results & patches
 *    Claude → (up, "improve") → re-runs the trace
 *
 * Two dotted vertical lines with opposing arrowheads + mono labels.
 */
const TraceLoopConnector: React.FC = () => {
  return (
    <div className={styles.wrap} aria-hidden>
      <span className={`${styles.line} ${styles.lineLeft}`}>
        <span className={styles.arrowDown} />
      </span>
      <span className={`${styles.line} ${styles.lineRight}`}>
        <span className={styles.arrowUp} />
      </span>
    </div>
  );
};


export default TraceLoopConnector;
