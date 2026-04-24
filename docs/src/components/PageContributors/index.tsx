import { Users } from "lucide-react";
import type { Contributor } from "@/lib/contributors";
import ContributorDisplay from "@/src/components/ContributorDisplay";
import ContributorsOverflow from "./ContributorsOverflow";
import styles from "./PageContributors.module.scss";

// How many avatars we show before collapsing the rest into a `+N`
// overflow pill. The manifest keeps the full list — this is purely a
// presentational cap. Picked 5 to match the visual density of
// GitHub's own contributor summary on repo pages.
const DEFAULT_LIMIT = 5;

interface PageContributorsProps {
  contributors: Contributor[];
  limit?: number;
}

function commitLabel(n: number) {
  return `${n} commit${n === 1 ? "" : "s"}`;
}

// Some resolved contributors can lack a meaningful commit count label
// (for example, if attribution metadata was backfilled from a known
// identity rather than a resolved GitHub commit). In that case, fall
// back to just the name.
function contributorLabel(c: Contributor) {
  return c.commits > 0 ? `${c.name} — ${commitLabel(c.commits)}` : c.name;
}

/**
 * Compact, avatar-only strip rendered inside the TOC footer on docs
 * pages. Each avatar is a link to the committer's GitHub profile; the
 * name surfaces on hover via `title` (native tooltip) + aria-label
 * (screen readers).
 *
 * If more than `limit` contributors exist, the overflow collapses into
 * a non-interactive `+N` pill whose `title` lists the hidden names.
 * Server-only; no client JS.
 */
const PageContributors: React.FC<PageContributorsProps> = ({
  contributors,
  limit = DEFAULT_LIMIT,
}) => {
  if (contributors.length === 0) return null;

  const shown = contributors.slice(0, limit);
  const overflow = contributors.slice(limit);

  return (
    <aside className={styles.wrapper} aria-label="Contributors to this page">
      {/* Heading mirrors fumadocs' own "On this page" TOC title:
       * same <h3> + classes, plus `data-toc-heading` so our scoped
       * rule in `app/global.css` (`#nd-toc [data-toc-heading]`)
       * pulls it up to 13px/dark-foreground to match `#toc-title`.
       * Only the icon differs — `Users` instead of `Text`. */}
      <h3
        data-toc-heading
        className="inline-flex items-center gap-1.5 text-sm text-fd-muted-foreground"
      >
        <Users className="size-4" aria-hidden="true" />
        <span>Contributors</span>
      </h3>
      <ul className={styles.list}>
        {shown.map((c) => (
          <li key={c.login} className={styles.item}>
            <ContributorDisplay
              href={c.url}
              avatarUrl={c.avatarUrl}
              label={contributorLabel(c)}
              title={contributorLabel(c)}
              tooltip={contributorLabel(c)}
            />
          </li>
        ))}
        {overflow.length > 0 ? (
          <li className={styles.item}>
            <ContributorsOverflow contributors={overflow} />
          </li>
        ) : null}
      </ul>
    </aside>
  );
};


export default PageContributors;
