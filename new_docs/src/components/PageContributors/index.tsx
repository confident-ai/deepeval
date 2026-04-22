import { Users } from "lucide-react";
import type { Contributor } from "@/lib/contributors";
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

// Pinned cofounders appear on every page with `commits: 0` even when
// they've never touched that file. In that case, the tooltip/aria label
// would read "Jeffrey Ip — 0 commits", which is both wrong-feeling and
// ugly. Fall back to just the name.
function contributorLabel(c: Contributor) {
  return c.commits > 0 ? `${c.name} — ${commitLabel(c.commits)}` : c.name;
}

/**
 * Compact, avatar-only strip rendered below `<PageLastUpdate>` on docs
 * pages. Each avatar is a link to the committer's GitHub profile; the
 * name surfaces on hover via `title` (native tooltip) + aria-label
 * (screen readers).
 *
 * If more than `limit` contributors exist, the overflow collapses into
 * a non-interactive `+N` pill whose `title` lists the hidden names.
 * Server-only; no client JS.
 */
export default function PageContributors({
  contributors,
  limit = DEFAULT_LIMIT,
}: PageContributorsProps) {
  if (contributors.length === 0) return null;

  const shown = contributors.slice(0, limit);
  const overflow = contributors.slice(limit);

  return (
    // `data-toc-full-bleed` opts this section out of the `#nd-toc`
    // inline-padding rule so its `border-top` can span the full TOC
    // column width like a real section divider. We then re-introduce
    // the 0.75rem left indent on the inner content so the heading /
    // avatars still align with the TOC title above.
    //
    // Owning the divider + top padding here (rather than on an outer
    // wrapper in the consumer) keeps the meta/spacing concerns local
    // to the component: if `PageContributors` is hidden, its divider
    // and padding disappear with it — the consumer doesn't end up
    // with an orphaned empty border.
    <aside
      data-toc-full-bleed
      className={`${styles.wrapper} mt-4 pt-4 border-t pl-3`}
      aria-label="Contributors to this page"
    >
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
            <a
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              title={contributorLabel(c)}
              aria-label={contributorLabel(c)}
              className={styles.link}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={c.avatarUrl}
                alt=""
                className={styles.avatar}
                width={24}
                height={24}
                loading="lazy"
              />
            </a>
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
}
