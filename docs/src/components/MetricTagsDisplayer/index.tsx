import React from "react";
import { TriangleAlert } from "lucide-react";
import styles from "./MetricTagsDisplayer.module.scss";

interface MetricTagsDisplayerProps {
  usesLLMs?: boolean;
  // Jev (TypeSafe AI's System One model) can judge this metric: either as
  // the only judge (JevEval, pass usesLLMs={false}) or as an alternative to
  // the LLM under the `hybrid` / `system_one` eval modes, in which case both
  // judge pills show.
  jev?: boolean;
  // Drex (Nace's decision models) is the judge: DrexEval. Drex Max reads
  // images, so unlike Jev it leaves the metric multimodal.
  drex?: boolean;
  singleTurn?: boolean;
  multiTurn?: boolean;
  referenceless?: boolean;
  referenceBased?: boolean;
  rag?: boolean;
  agent?: boolean;
  trajectory?: boolean;
  chatbot?: boolean;
  custom?: boolean;
  safety?: boolean;
  multimodal?: boolean;
  community?: boolean;
}

const MetricTagsDisplayer = ({
  usesLLMs = true,
  jev = false,
  drex = false,
  singleTurn = false,
  multiTurn = false,
  referenceless = false,
  referenceBased = false,
  rag = false,
  agent = false,
  trajectory = false,
  chatbot = false,
  custom = false,
  safety = false,
  multimodal = true,
  community = false,
}: MetricTagsDisplayerProps) => {
  // Jev is text-only, so a metric with no LLM judge cannot be multimodal,
  // unless Drex (Drex Max reads images) is the judge.
  if (!usesLLMs && !drex) multimodal = false;

  return (
    <div className={styles.metricTagsDisplayer}>
      {community && (
        <div className={`${styles.pill} ${styles.community}`}>
          <TriangleAlert aria-hidden="true" />
          <span>Community</span>
        </div>
      )}
      {usesLLMs && (
        <div className={`${styles.pill} ${styles.usesLLM}`}>LLM-as-a-judge</div>
      )}
      {jev && (
        <div className={`${styles.pill} ${styles.jev}`}>Jev-as-a-judge</div>
      )}
      {drex && (
        <div className={`${styles.pill} ${styles.drex}`}>Drex-as-a-judge</div>
      )}
      {custom && (
        <div className={`${styles.pill} ${styles.custom}`}>Custom</div>
      )}
      {singleTurn && (
        <div className={`${styles.pill} ${styles.singleTurn}`}>Single-turn</div>
      )}
      {multiTurn && (
        <div className={`${styles.pill} ${styles.multiTurn}`}>Multi-turn</div>
      )}
      {trajectory && (
        <div className={`${styles.pill} ${styles.trajectory}`}>Trajectory</div>
      )}
      {referenceless && (
        <div className={`${styles.pill} ${styles.referenceless}`}>
          Referenceless
        </div>
      )}
      {referenceBased && (
        <div className={`${styles.pill} ${styles.referenceBased}`}>
          Reference-based
        </div>
      )}
      {rag && <div className={`${styles.pill} ${styles.rag}`}>RAG</div>}
      {agent && <div className={`${styles.pill} ${styles.agent}`}>Agent</div>}
      {chatbot && (
        <div className={`${styles.pill} ${styles.chatbot}`}>Chatbot</div>
      )}
      {safety && (
        <div className={`${styles.pill} ${styles.safety}`}>Safety</div>
      )}
      {multimodal && (
        <div className={`${styles.pill} ${styles.multimodal}`}>Multimodal</div>
      )}
    </div>
  );
};

export default MetricTagsDisplayer;
