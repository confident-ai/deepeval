import React from "react";
import styles from "./IntegrationTagsDisplayer.module.scss";

interface IntegrationTagsDisplayerProps {
  otel?: boolean;
  native?: boolean;
  cicdEvals?: boolean;
  traceability?: boolean;
}

const IntegrationTagsDisplayer = ({
  otel = false,
  native = false,
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
      {native && (
        <div className={`${styles.pill} ${styles.native}`}>
          Native Instrumentation
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
