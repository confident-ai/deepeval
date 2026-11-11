import {
  formatMetric,
  sortEntries,
  type Benchmark,
} from "@/lib/benchmarks";
import ModelName from "../ModelName";
import ProviderMark from "../ProviderMark";
import { Trophy, RankBadge } from "../RankTrophy";
import styles from "./BestPicks.module.scss";

type BestPicksProps = {
  benchmark: Benchmark;
};

const RUNNERS_UP = 4;

const BestPicks: React.FC<BestPicksProps> = ({ benchmark }) => {
  const picks = benchmark.metrics.filter((metric) => metric.pickLabel);

  return (
    <div className={styles.grid}>
      {picks.map((metric) => {
        const [winner, ...rest] = sortEntries(benchmark.entries, metric);
        return (
          <article key={metric.id} className={styles.card}>
            <p className={styles.label}>{metric.pickLabel}</p>
            <div className={styles.winner}>
              <ProviderMark provider={winner.provider} />
              <div className={styles.winnerText}>
                <h3 className={styles.winnerModel}>
                  {winner.model}
                  <Trophy rank={1} />
                </h3>
                <p className={styles.winnerValue}>
                  <span>{formatMetric(winner.scores[metric.id], metric)}</span>{" "}
                  {metric.label.toLowerCase()}
                </p>
              </div>
            </div>
            <ol className={styles.list} start={2}>
              {rest.slice(0, RUNNERS_UP).map((entry, index) => (
                <li key={entry.model} className={styles.row}>
                  <span className={styles.rank}>
                    <RankBadge rank={index + 2} />
                  </span>
                  <span className={styles.model}>
                    <ModelName model={entry.model} provider={entry.provider} />
                  </span>
                  <span className={styles.value}>
                    {formatMetric(entry.scores[metric.id], metric)}
                  </span>
                </li>
              ))}
            </ol>
          </article>
        );
      })}
    </div>
  );
};

export default BestPicks;
