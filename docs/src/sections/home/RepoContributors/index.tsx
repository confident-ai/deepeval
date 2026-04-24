import Link from "next/link";
import contributors from "@site/lib/generated/repo-contributors.json";
import { gitConfig } from "@site/lib/shared";
import ContributorDisplay from "@/src/components/ContributorDisplay";
import styles from "./RepoContributors.module.scss";

interface RepoContributor {
  login: string;
  avatarUrl: string;
  url: string;
  contributions: number;
}

const list = contributors as RepoContributor[];

interface RepoContributorsProps {
  /**
   * Maximum avatars to render. Defaults to "all of them" — at 32px the
   * full 250+ wall still only takes ~12 rows, and showing everyone is
   * the whole point of this section. If a `limit` is passed, the
   * remainder collapses into a "+N more" link to the repo's
   * contributors page.
   */
  limit?: number;
}

function commitsLabel(n: number) {
  return `${n.toLocaleString()} contribution${n === 1 ? "" : "s"}`;
}

function contributorLabel(c: RepoContributor) {
  return `${c.login} — ${commitsLabel(c.contributions)}`;
}

const RepoContributors: React.FC<RepoContributorsProps> = ({ limit }) => {
  if (list.length === 0) return null;

  const cap = limit ?? list.length;
  const shown = list.slice(0, cap);
  const overflow = Math.max(0, list.length - shown.length);
  const repoContribsUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}/graphs/contributors`;

  return (
    <section
      className={styles.wrapper}
      aria-label={`${list.length} contributors to ${gitConfig.repo}`}
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
            aria-label={`See all ${list.length} contributors on GitHub`}
            title={`See all ${list.length} contributors on GitHub`}
          >
            +{overflow}
          </Link>
        ) : null}
      </div>
    </section>
  );
};


export default RepoContributors;
