import Link from "next/link";
import ProviderMark from "../ProviderMark";
import { Trophy } from "../RankTrophy";
import styles from "./WinnersCarousel.module.scss";

export type Winner = {
  slug: string;
  benchmark: string;
  metric: string;
  score: string;
  model: string;
  provider: string;
};

type WinnersCarouselProps = {
  winners: Winner[];
};

// Seconds each row takes to pass, so the speed stays the same however many
// benchmarks there are.
const SECONDS_PER_ROW = 3.2;

const WinnerRow: React.FC<{ winner: Winner; hidden?: boolean }> = ({
  winner,
  hidden = false,
}) => (
  <li className={styles.row} aria-hidden={hidden || undefined}>
    <Link
      href={`/benchmarks/${winner.slug}`}
      className={styles.link}
      tabIndex={hidden ? -1 : undefined}
      aria-label={`${winner.benchmark}: led by ${winner.model} at ${winner.score} ${winner.metric}`}
    >
      <ProviderMark provider={winner.provider} size="lg" />
      <span className={styles.text}>
        <span className={styles.benchmark}>{winner.benchmark}</span>
        <span className={styles.model}>
          <Trophy rank={1} size={14} />
          {winner.model}
          <span className={styles.metric}>{winner.metric}</span>
        </span>
      </span>
      <span className={styles.score}>{winner.score}</span>
    </Link>
  </li>
);

const WinnersCarousel: React.FC<WinnersCarouselProps> = ({ winners }) => {
  return (
    <div
      className={styles.carousel}
      aria-label="Top-ranked model on each benchmark"
    >
      {/* The list is rendered twice and shifted by half its height, which
          makes the loop seamless. The second copy is hidden from assistive
          technology. */}
      <ul
        className={styles.track}
        style={{
          ["--duration" as string]: `${winners.length * SECONDS_PER_ROW}s`,
        }}
      >
        {winners.map((winner) => (
          <WinnerRow key={winner.slug} winner={winner} />
        ))}
        {winners.map((winner) => (
          <WinnerRow key={`${winner.slug}-repeat`} winner={winner} hidden />
        ))}
      </ul>
    </div>
  );
};

export default WinnersCarousel;
