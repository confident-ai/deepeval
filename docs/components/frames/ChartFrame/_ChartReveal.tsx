"use client";

import type * as React from "react";
import { useInView } from "react-intersection-observer";
import styles from "./styles.module.scss";

export const ChartReveal = ({ children }: { children: React.ReactNode }): React.ReactElement => {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.3 });
  return (
    <div ref={ref} className={styles.reveal} data-revealed={inView}>
      {children}
    </div>
  );
};
