import { formatRank } from "@/lib/benchmarks/format";
import styles from "./RankTrophy.module.scss";

const TROPHY_NAMES: Record<number, string> = {
  1: "Gold",
  2: "Silver",
  3: "Bronze",
};

export const hasTrophy = (rank: number) => rank >= 1 && rank <= 3;

type TrophyProps = {
  rank: number;
  size?: number;
};

// Gold, silver or bronze for ranks 1 to 3; renders nothing below that.
export const Trophy: React.FC<TrophyProps> = ({ rank, size = 16 }) => {
  if (!hasTrophy(rank)) return null;

  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      className={styles.trophy}
      data-rank={rank}
      role="img"
      aria-label={`${TROPHY_NAMES[rank]} trophy`}
    >
      <path
        d="M4.2 3H2.1v1.3a2.6 2.6 0 0 0 2.6 2.6M11.8 3h2.1v1.3a2.6 2.6 0 0 1-2.6 2.6"
        className={styles.handles}
      />
      <path d="M4 1.5h8v4a4 4 0 0 1-8 0z" className={styles.cup} />
      <path d="M7.1 9.2h1.8v2.6H7.1z" className={styles.stem} />
      <path d="M4.6 11.8h6.8v2.4H4.6z" className={styles.cup} />
    </svg>
  );
};

type RankBadgeProps = {
  rank: number;
};

// The top three show a trophy in place of the number; everything below
// shows the two-digit rank. Both occupy the same width.
export const RankBadge: React.FC<RankBadgeProps> = ({ rank }) => {
  return (
    <span className={styles.badge}>
      {hasTrophy(rank) ? (
        <>
          <Trophy rank={rank} />
          <span className={styles.srOnly}>{formatRank(rank)}</span>
        </>
      ) : (
        formatRank(rank)
      )}
    </span>
  );
};

export default Trophy;
