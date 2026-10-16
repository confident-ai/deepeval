import { Check, X } from "lucide-react";
import styles from "./EnterpriseComparisonTable.module.scss";

const ROWS = [
  {
    confident: "Failing production traces surface on their own",
    deepeval: "You write the eval before you can catch the failure",
  },
  {
    confident: "Failing traces clustered into failure modes",
    deepeval: "Failures reviewed one test case at a time",
  },
  {
    confident: "Failing traces become goldens for the next eval",
    deepeval: "Goldens curated by hand",
  },
  {
    confident: "Evals on every production trace",
    deepeval: "Evals in development and CI/CD",
  },
  {
    confident: "Traces routed to the right annotators",
    deepeval: "Developer-mediated annotation",
  },
  {
    confident: "AI governance as release gates",
    deepeval: "Pass/fail per test suite",
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
