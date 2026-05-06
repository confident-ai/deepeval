import React from "react";
import styles from "./IntegrationTagsDisplayer.module.scss";

interface IntegrationTagsDisplayerProps {
  otel?: boolean;
  manual?: boolean;
  cicdEvals?: boolean;
  traceability?: boolean;
}

const IntegrationTagsDisplayer = ({
  otel = false,
  manual = false,
  cicdEvals = false,
  traceability = false,
}: IntegrationTagsDisplayerProps) => {
  return (
    <div className={styles.integrationTagsDisplayer}>
      {otel && (
        <div className={`${styles.pill} ${styles.otel}`}>
          OTel Instrumentation
        </div>
      )}
      {manual && (
        <div className={`${styles.pill} ${styles.manual}`}>
          Manual instrumentation
        </div>
      )}
      {cicdEvals && (
        <div className={`${styles.pill} ${styles.cicdEvals}`}>
          Evals in CI/CD
        </div>
      )}
      {traceability && (
        <div className={`${styles.pill} ${styles.traceability}`}>
          Evals with Traceability
        </div>
      )}
    </div>
  );
};

export default IntegrationTagsDisplayer;
