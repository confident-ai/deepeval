import React from "react";
import styles from "./GuardTagsDisplayer.module.scss";

interface GuardTagsDisplayerProps {
  singleTurn?: boolean;
  usesLLMs?: boolean;
  input?: boolean;
  output?: boolean;
}

const GuardTagsDisplayer = ({
  singleTurn = false,
  usesLLMs = false,
  input = false,
  output = false,
}: GuardTagsDisplayerProps) => {
  return (
    <div className={styles.tagsDisplayer}>
      {singleTurn && (
        <div className={`${styles.pill} ${styles.singleTurn}`}>Single-turn</div>
      )}
      {usesLLMs && (
        <div className={`${styles.pill} ${styles.usesLLM}`}>LLM-as-a-judge</div>
      )}
      {input && (
        <div className={`${styles.pill} ${styles.input}`}>Input guard</div>
      )}
      {output && (
        <div className={`${styles.pill} ${styles.output}`}>Output guard</div>
      )}
    </div>
  );
};

export default GuardTagsDisplayer;
