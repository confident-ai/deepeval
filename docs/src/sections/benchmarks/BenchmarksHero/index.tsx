import { ArrowRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@/src/components/Buttons";
import WinnersCarousel, { type Winner } from "../WinnersCarousel";
import styles from "./BenchmarksHero.module.scss";

export type HeroStat = {
  label: string;
  value: string;
};

type BenchmarksHeroProps = {
  stats: HeroStat[];
  winners: Winner[];
};

const BenchmarksHero: React.FC<BenchmarksHeroProps> = ({ stats, winners }) => {
  return (
    <section className={styles.hero}>
      <div className={styles.top} data-aside={winners.length > 0}>
        <div className={styles.main}>
        <p className={styles.eyebrow}>Model Benchmarks</p>
        <h1 className={styles.title}>
          Open benchmarks for generative AI.
        </h1>
        <p className={styles.description}>
          Public datasets, published grading and results you can reproduce.
          Every leaderboard on this page runs through DeepEval, so the command
          that produced a row is the same one you run on your own model.
        </p>
        <div className={styles.actions}>
          <PrimaryButton href="#benchmarks" endIcon={<ArrowRight aria-hidden />}>
            Browse benchmarks
          </PrimaryButton>
          <SecondaryButton href="/docs/benchmarks-introduction">
            Run one yourself
          </SecondaryButton>
        </div>
        </div>
        {winners.length > 0 && (
          <div className={styles.aside}>
            <WinnersCarousel winners={winners} />
          </div>
        )}
      </div>

      <dl className={styles.stats}>
        {stats.map((stat) => (
          <div key={stat.label} className={styles.stat}>
            <dt className={styles.statLabel}>{stat.label}</dt>
            <dd className={styles.statValue}>{stat.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
};

export default BenchmarksHero;
