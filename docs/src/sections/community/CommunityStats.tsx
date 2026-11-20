"use client";

import {
  formatStarCount,
  useGithubStarCount,
} from "@/src/components/GithubCtaButton/useGithubStarCount";
import styles from "./CommunityStats.module.scss";

type CommunityStatsProps = {
  contributors: number;
  contributions: number;
  events: number;
  cities: number;
};

const compact = (n: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

// Four sourced numbers. Stars load live from GitHub (cached for an hour,
// like the nav button); the rest come from the build.
const CommunityStats: React.FC<CommunityStatsProps> = ({
  contributors,
  contributions,
  events,
  cities,
}) => {
  const stars = useGithubStarCount();

  const tiles = [
    { value: stars === null ? "…" : formatStarCount(stars), label: "GitHub stars", source: "Live from GitHub", accent: "#f2c230" },
    { value: contributors.toLocaleString("en-US"), label: "Contributors", source: "GitHub contributors", accent: "#2ec4a6" },
    { value: compact(contributions), label: "Contributions", source: "Commits to the repo", accent: "#ff8a3d" },
    { value: `${events} / ${cities}`, label: "Meetups / cities", source: "Listed on this page", accent: "#9b5cff" },
  ];

  return (
    <div className={`not-prose ${styles.grid}`}>
      {tiles.map((t) => (
        <div key={t.label} className={styles.tile} style={{ "--accent": t.accent } as React.CSSProperties}>
          <span className={styles.value}>{t.value}</span>
          <span className={styles.label}>{t.label}</span>
          <span className={styles.source}>{t.source}</span>
        </div>
      ))}
    </div>
  );
};

export default CommunityStats;
