import { Check, X } from "lucide-react";
import styles from "./EnterpriseComparisonTable.module.scss";

const ROWS = [
  {
    confident: "One quality standard across every team",
    deepeval: "Each team's evals live in its own code",
  },
  {
    confident: "Centralized metrics",
    deepeval: "Metrics scattered in code",
  },
  {
    confident: "AI governance as release gates",
    deepeval: "Pass/fail per test suite",
  },
  {
    confident: "Evals on every production trace",
    deepeval: "Evals in development and CI/CD",
  },
  {
    confident: "Failing traces become goldens for the next eval",
    deepeval: "Goldens curated by hand",
  },
  {
    confident: "Traces routed to the right annotators",
    deepeval: "Developer-mediated annotation",
  },
  {
    confident: "Red teaming in the same system",
    deepeval: "Red teaming set up separately",
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
