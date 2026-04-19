import React, { ReactNode } from "react";
import styles from "./FAQ.module.scss";

export interface QA {
  question: string;
  answer: ReactNode;
}

interface FAQsProps {
  qas: QA[];
}

export function FAQs({ qas }: FAQsProps) {
  return (
    <div className={styles.list}>
      {qas.map(({ question, answer }) => (
        <details key={question} className={styles.item}>
          <summary className={styles.summary}>
            <span className={styles.question}>{question}</span>
            <span className={styles.icon} aria-hidden="true" />
          </summary>
          <div className={styles.answer}>{answer}</div>
        </details>
      ))}
    </div>
  );
}

export default FAQs;
