import Link from "next/link";
import contributors from "@/lib/generated/changelog-contributors.json";
import { gitConfig } from "@/lib/shared";
import ContributorDisplay from "@/src/components/ContributorDisplay";
import styles from "./ChangelogContributors.module.scss";

interface ChangelogContributor {
  login: string;
  name: string;
  url: string;
  avatarUrl: string;
  contributions: number;
}

type ChangelogContributorManifest = Record<string, ChangelogContributor[]>;

interface ChangelogContributorsProps {
  year: string | number;
  limit?: number;
}

const manifest = contributors as ChangelogContributorManifest;

function contributionsLabel(n: number) {
  return `${n.toLocaleString()} changelog entr${n === 1 ? "y" : "ies"}`;
}

function contributorLabel(c: ChangelogContributor) {
  return `${c.name} — ${contributionsLabel(c.contributions)}`;
}

const ChangelogContributors: React.FC<ChangelogContributorsProps> = ({
  year,
  limit,
}) => {
  const list = manifest[String(year)] ?? [];
  if (list.length === 0) return null;

  const cap = limit ?? list.length;
  const shown = list.slice(0, cap);
  const overflow = Math.max(0, list.length - shown.length);
  const repoContribsUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}/graphs/contributors`;

  return (
    <section
      className={styles.wrapper}
      aria-label={`${list.length} contributors in ${year}`}
    >
      <div className={styles.grid}>
        {shown.map((c) => (
          <ContributorDisplay
            key={c.login}
            href={c.url}
            avatarUrl={c.avatarUrl}
            label={contributorLabel(c)}
            tooltip={contributorLabel(c)}
            size="md"
          />
        ))}
        {overflow > 0 ? (
          <Link
            href={repoContribsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.overflow}
            aria-label={`See all ${list.length} DeepEval contributors on GitHub`}
            title={`See all ${list.length} DeepEval contributors on GitHub`}
          >
            +{overflow}
          </Link>
        ) : null}
      </div>
    </section>
  );
};

export default ChangelogContributors;
