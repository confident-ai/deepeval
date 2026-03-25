import React, { ReactNode } from "react";
import MDXContent from "@theme/MDXContent";
import styles from "./index.module.css";

interface TimelineProps {
  children: ReactNode;
}

export function Timeline({ children }: TimelineProps) {
  return <div className={styles.timeline}>{children}</div>;
}

interface TimelineItemProps {
  title: string;
  children: ReactNode;
}

export function TimelineItem({ title, children }: TimelineItemProps) {
  return (
    <div className={styles.timelineItem}>
      <div className={styles.timelineStep}>
        <div className={styles.timelineNumber}></div>
        <div className={styles.timelineLine}></div>
      </div>
      <div className={styles.timelineContent}>
        <h3 className={styles.timelineTitle}>{title}</h3>
        <MDXContent>{children}</MDXContent>
      </div>
    </div>
  );
}
