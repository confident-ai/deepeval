import React from "react";
import styles from "./MetricTagsDisplayer.module.css";

const MetricTagsDisplayer = ({ usesLLMs=true, referenceless=false, referenceBased=false, rag=false, agent=false, chatbot=false, custom=false, safety=false, multimodal=false }) => {
  return (
    <div className={styles.metricTagsDisplayer}>
      {usesLLMs && <div className={`${styles.pill} ${styles.usesLLM}`}>{multimodal ? "M" : ""}LLM-as-a-judge</div>}
      {referenceless && <div className={`${styles.pill} ${styles.referenceless}`}>Referenceless metric</div>}
      {referenceBased && <div className={`${styles.pill} ${styles.referenceBased}`}>Reference-based metric</div>}
      {rag && <div className={`${styles.pill} ${styles.rag}`}>RAG metric</div>}
      {agent && <div className={`${styles.pill} ${styles.agent}`}>Agent metric</div>}
      {chatbot && <div className={`${styles.pill} ${styles.chatbot}`}>Chatbot metric</div>}
      {custom && <div className={`${styles.pill} ${styles.custom}`}>Custom metric</div>}
      {safety && <div className={`${styles.pill} ${styles.safety}`}>Safety metric</div>}
      {multimodal && <div className={`${styles.pill} ${styles.multimodal}`}>Multimodal</div>}
    </div>
  );
};

export default MetricTagsDisplayer;
