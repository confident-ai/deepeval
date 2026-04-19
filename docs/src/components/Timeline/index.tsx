import React, { ReactNode } from "react";
import MDXContent from "@theme/MDXContent";
import styles from "./index.module.scss";

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
    <div className={styles.item}>
      <div className={styles.step}>
        <div className={styles.number}></div>
        <div className={styles.line}></div>
      </div>
      <div className={styles.content}>
        <h3 className={styles.title}>{title}</h3>
        <MDXContent>{children}</MDXContent>
      </div>
    </div>
  );
}
