import React, { ReactNode } from "react";
import styles from "./FAQ.module.scss";
import SchemaInjector from "../SchemaInjector/SchemaInjector";
import { buildFAQPageSchema } from "@site/src/utils/schema-helpers";

export interface QA {
  question: string;
  answer: ReactNode;
}

interface FAQsProps {
  qas: QA[];
}

function extractText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (React.isValidElement(node)) {
    return extractText((node.props as { children?: ReactNode }).children);
  }
  return "";
}

export function FAQs({ qas }: FAQsProps) {
  const schema = buildFAQPageSchema(
    qas.map(({ question, answer }) => ({
      question,
      answer: extractText(answer).replace(/\s+/g, " ").trim(),
    }))
  );

  return (
    <>
      <SchemaInjector schema={schema} />
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
    </>
  );
}

export default FAQs;
