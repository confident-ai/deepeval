import React from "react";
import styles from "./index.module.css";

export function Timeline({ children }) {
  const childrenWithNumbers = React.Children.map(children, (child, index) => {
    return React.cloneElement(child, {
      number: index + 1,
    });
  });
  return <div className={styles.timeline}>{childrenWithNumbers}</div>;
}

export function TimelineItem({ title, children, number }) {
  return (
    <div className={styles.timelineItem}>
      <div className={styles.timelineStep}>
        <div className={styles.timelineNumber}>{number}</div>
        <div className={styles.timelineLine}></div>
      </div>
      <div className={styles.timelineContent}>
        <h3>{title}</h3>
        {children}
      </div>
    </div>
  );
}
