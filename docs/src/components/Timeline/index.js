import React from "react";
import MDXContent from "@theme/MDXContent";
import styles from "./index.module.css";

export function Timeline({ children }) {
  return <div className={styles.timeline}>{children}</div>;
}

export function TimelineItem({ title, children }) {
  return (
    <div className={styles.timelineItem}>
      <div className={styles.timelineStep}>
        <div className={styles.timelineNumber}></div>
        <div className={styles.timelineLine}></div>
      </div>
      <div className={styles.timelineContent}>
        <h3>{title}</h3>
        <MDXContent>{children}</MDXContent>
      </div>
    </div>
  );
}
