import Link from "next/link";
import { ArrowLeft, ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@/src/components/Buttons";
import { externalRelForOutboundHref } from "@/src/utils/outbound-link-rel";
import {
  GRADING_LABELS,
  formatCount,
  formatDate,
  formatMetric,
  formatRank,
  getMetric,
  rankEntries,
  type Benchmark,
} from "@/lib/benchmarks";
import { ProviderBackdrop } from "../ProviderMark";
import { Trophy } from "../RankTrophy";
import styles from "./BenchmarkHeader.module.scss";

type BenchmarkHeaderProps = {
  benchmark: Benchmark;
};

const BenchmarkHeader: React.FC<BenchmarkHeaderProps> = ({ benchmark }) => {
  const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
  const podium = rankEntries(benchmark).slice(0, 3);

  const facts = [
    { label: "Version", value: benchmark.version },
    { label: "Updated", value: formatDate(benchmark.updatedAt) },
    { label: "Models", value: String(benchmark.entries.length) },
    { label: "Grading", value: GRADING_LABELS[benchmark.grading.method] },
    {
      label: "Dataset",
      value: `${formatCount(benchmark.dataset.size)} ${benchmark.dataset.unit}`,
    },
  ];

  return (
    <header className={styles.header}>
      <div className={styles.main}>
        <Link href="/benchmarks" className={styles.back}>
          <ArrowLeft aria-hidden />
          All benchmarks
        </Link>

        <h1 className={styles.title}>{benchmark.name}</h1>
        <p className={styles.summary}>{benchmark.summary}</p>

        <div className={styles.actions}>
          <PrimaryButton href="#run">Run with DeepEval</PrimaryButton>
          <SecondaryButton
            href={benchmark.dataset.url}
            target="_blank"
            rel={externalRelForOutboundHref(benchmark.dataset.url)}
            endIcon={<ArrowUpRight aria-hidden />}
          >
            Dataset on {benchmark.dataset.host}
          </SecondaryButton>
        </div>
      </div>

      <ol className={styles.podium} aria-label={`Top 3 by ${primary.label}`}>
        {podium.map((entry) => (
          <li key={entry.model} className={styles.podiumItem}>
            <ProviderBackdrop
              provider={entry.provider}
              className={styles.podiumBackdrop}
            />
            <span className={styles.podiumRank}>
              {formatRank(entry.rank)}
            </span>
            <span className={styles.podiumScore}>
              {formatMetric(entry.scores[primary.id], primary)}
            </span>
            <span className={styles.podiumModel}>
              <span>{entry.model}</span>
              <Trophy rank={entry.rank} />
            </span>
            <span className={styles.podiumProvider}>{entry.provider}</span>
          </li>
        ))}
      </ol>

      <dl className={styles.facts}>
        {facts.map((fact) => (
          <div key={fact.label} className={styles.fact}>
            <dt>{fact.label}</dt>
            <dd>{fact.value}</dd>
          </div>
        ))}
      </dl>
    </header>
  );
};

export default BenchmarkHeader;
