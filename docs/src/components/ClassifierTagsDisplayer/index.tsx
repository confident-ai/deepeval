import React from "react";
import styles from "./ClassifierTagsDisplayer.module.scss";

// Tag pills shown under each classifier's H1, mirroring MetricTagsDisplayer.
// Every classifier is a categorical LLM-as-a-judge, so those two pills are
// always on; `jev` marks that Jev can decide the label instead under the
// `system_one` eval mode; the rest describe scope and use case.
interface ClassifierTagsDisplayerProps {
  jev?: boolean;
  custom?: boolean;
  singleTurn?: boolean;
  multiTurn?: boolean;
  safety?: boolean;
  security?: boolean;
  policy?: boolean;
  rag?: boolean;
  agent?: boolean;
  chatbot?: boolean;
  format?: boolean;
}

const ClassifierTagsDisplayer = ({
  jev = false,
  custom = false,
  singleTurn = false,
  multiTurn = false,
  safety = false,
  security = false,
  policy = false,
  rag = false,
  agent = false,
  chatbot = false,
  format = false,
}: ClassifierTagsDisplayerProps) => {
  return (
    <div className={styles.classifierTagsDisplayer}>
      <div className={`${styles.pill} ${styles.usesLLM}`}>LLM-as-a-judge</div>
      {jev && (
        <div className={`${styles.pill} ${styles.jev}`}>Jev-as-a-judge</div>
      )}
      <div className={`${styles.pill} ${styles.categorical}`}>Categorical</div>
      {custom && (
        <div className={`${styles.pill} ${styles.custom}`}>Custom</div>
      )}
      {singleTurn && (
        <div className={`${styles.pill} ${styles.singleTurn}`}>Single-turn</div>
      )}
      {multiTurn && (
        <div className={`${styles.pill} ${styles.multiTurn}`}>Multi-turn</div>
      )}
      {safety && (
        <div className={`${styles.pill} ${styles.safety}`}>Safety</div>
      )}
      {security && (
        <div className={`${styles.pill} ${styles.security}`}>Security</div>
      )}
      {policy && (
        <div className={`${styles.pill} ${styles.policy}`}>Policy</div>
      )}
      {rag && <div className={`${styles.pill} ${styles.rag}`}>RAG</div>}
      {agent && <div className={`${styles.pill} ${styles.agent}`}>Agent</div>}
      {chatbot && (
        <div className={`${styles.pill} ${styles.chatbot}`}>Chatbot</div>
      )}
      {format && (
        <div className={`${styles.pill} ${styles.format}`}>Format</div>
      )}
    </div>
  );
};

export default ClassifierTagsDisplayer;
