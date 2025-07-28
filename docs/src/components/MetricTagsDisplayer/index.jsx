import React from "react";
import styles from "./MetricTagsDisplayer.module.css";

const MetricTagsDisplayer = ({ usesLLMs=true, singleTurn=false, multiTurn=false,referenceless=false, referenceBased=false, rag=false, agent=false, chatbot=false, custom=false, safety=false, multimodal=false }) => {
  return (
    <div className={styles.metricTagsDisplayer}>
      {usesLLMs && <div className={`${styles.pill} ${styles.usesLLM}`}>{multimodal ? "M" : ""}LLM-as-a-judge</div>}
      {custom && <div className={`${styles.pill} ${styles.custom}`}>Custom</div>}
      {singleTurn && <div className={`${styles.pill} ${styles.singleTurn}`}>Single-turn</div>}
      {multiTurn && <div className={`${styles.pill} ${styles.multiTurn}`}>Multi-turn</div>}
      {referenceless && <div className={`${styles.pill} ${styles.referenceless}`}>Referenceless</div>}
      {referenceBased && <div className={`${styles.pill} ${styles.referenceBased}`}>Reference-based</div>}
      {rag && <div className={`${styles.pill} ${styles.rag}`}>RAG</div>}
      {agent && <div className={`${styles.pill} ${styles.agent}`}>Agent</div>}
      {chatbot && <div className={`${styles.pill} ${styles.chatbot}`}>Chatbot</div>}
      {safety && <div className={`${styles.pill} ${styles.safety}`}>Safety</div>}
      {multimodal && <div className={`${styles.pill} ${styles.multimodal}`}>Multimodal</div>}
    </div>
  );
};

export default MetricTagsDisplayer;
