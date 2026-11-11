import Link from "next/link";
import {
  CATEGORY_LABELS,
  GRADING_LABELS,
  formatCount,
  formatDate,
  formatMetric,
  getMetric,
  rankEntries,
  scoreFraction,
} from "@/lib/benchmarks/format";
import type { Benchmark } from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import { RankBadge } from "../RankTrophy";
import styles from "./BenchmarkCards.module.scss";

type BenchmarkCardsProps = {
  benchmarks: Benchmark[];
};

const TOP_N = 3;

const BenchmarkCard: React.FC<{ benchmark: Benchmark }> = ({ benchmark }) => {
  const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
  const ranked = rankEntries(benchmark);
  const values = ranked.map((entry) => entry.scores[primary.id]);

  return (
    <Link
      href={`/benchmarks/${benchmark.slug}`}
      className={styles.card}
      data-callout
    >
      <svg viewBox="0 0 24 24" className={styles.arrow} aria-hidden>
        <path d="M5 19 19 5M8 5h11v11" />
      </svg>

      <span className={styles.category}>
        {CATEGORY_LABELS[benchmark.category]}
      </span>

      <h3 className={styles.name}>{benchmark.name}</h3>
      <p className={styles.tagline}>{benchmark.tagline}</p>

      <ol className={styles.leaders} aria-label={`Top ${TOP_N} by ${primary.label}`}>
        {ranked.slice(0, TOP_N).map((entry) => (
          <li key={entry.model} className={styles.leader}>
            <span className={styles.leaderRank}>
              <RankBadge rank={entry.rank} />
            </span>
            <span className={styles.leaderModel}>
              <ModelName model={entry.model} provider={entry.provider} />
            </span>
            <span className={styles.leaderScore}>
              {formatMetric(entry.scores[primary.id], primary)}
            </span>
            <span className={styles.leaderTrack} aria-hidden>
              <span
                className={styles.leaderFill}
                style={{
                  width: `${scoreFraction(entry.scores[primary.id], primary, values) * 100}%`,
                }}
              />
            </span>
          </li>
        ))}
      </ol>

      <dl className={styles.meta}>
        <div>
          <dt>Grading</dt>
          <dd>{GRADING_LABELS[benchmark.grading.method]}</dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>
            {formatCount(benchmark.dataset.size)} {benchmark.dataset.unit}
          </dd>
        </div>
        <div>
          <dt>Updated</dt>
          <dd>{formatDate(benchmark.updatedAt)}</dd>
        </div>
      </dl>
    </Link>
  );
};

const BenchmarkCards: React.FC<BenchmarkCardsProps> = ({ benchmarks }) => {
  return (
    <div className={styles.grid}>
      {benchmarks.map((benchmark) => (
        <BenchmarkCard key={benchmark.slug} benchmark={benchmark} />
      ))}
    </div>
  );
};

export default BenchmarkCards;
