import { Check, X } from "lucide-react";
import styles from "./EnterpriseComparisonTable.module.scss";

const ROWS = [
  {
    confident: "Shared evaluation workspace",
    deepeval: "Testing results live in local files",
  },
  {
    confident: "No-code eval workflows",
    deepeval: "Local and CI/CD test runner",
  },
  {
    confident: "Production observability + tracing",
    deepeval: "Limited to pre-production testing",
  },
  {
    confident: "Online eval monitoring",
    deepeval: "Bring your own eval infra",
  },
  {
    confident: "Managed regression workflows",
    deepeval: "Engineer-owned test suites",
  },
  {
    confident: "Centralized metrics",
    deepeval: "Metrics scattered in code",
  },
  {
    confident: "Annotation queues for SMEs",
    deepeval: "Developer-mediated annotation",
  },
  {
    confident: "Enterprise controls",
    deepeval: "Single-user by design",
  },
];

const EnterpriseComparisonTable: React.FC = () => {
  return (
    <div
      className={styles.table}
      role="table"
      aria-label="DeepEval and Confident AI comparison"
    >
      <div className={styles.header} role="row">
        <div role="columnheader">Confident AI</div>
        <div role="columnheader">DeepEval</div>
      </div>
      <div className={styles.body}>
        {ROWS.map((row) => (
          <div key={row.confident} className={styles.row} role="row">
            <div className={styles.item} role="cell">
              <span className={styles.icon} data-tone="positive" aria-hidden>
                <Check />
              </span>
              <strong>{row.confident}</strong>
            </div>
            <div className={styles.item} role="cell">
              <span className={styles.icon} data-tone="negative" aria-hidden>
                <X />
              </span>
              <strong>{row.deepeval}</strong>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EnterpriseComparisonTable;
