"use client";

import Link from "next/link";

// The rest of <PageContributors> is a server component — only the
// popover trigger needs client interactivity, so we keep this splinter
// minimal. Radix Popover gives us keyboard/focus/outside-click for
// free, which would be a lot of wiring to rebuild by hand.

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "fumadocs-ui/components/ui/popover";
import type { Contributor } from "@/lib/contributors";
import styles from "./PageContributors.module.scss";

interface ContributorsOverflowProps {
  contributors: Contributor[];
}

const ContributorsOverflow: React.FC<ContributorsOverflowProps> = ({
  contributors,
}) => {
  if (contributors.length === 0) return null;
  return (
    <Popover>
      <PopoverTrigger
        className={styles.overflow}
        aria-label={`Show ${contributors.length} more contributor${contributors.length === 1 ? "" : "s"}`}
      >
        +{contributors.length}
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={6}
        className={styles.popover}
        // Radix gives the popover `aria-labelledby` pointing at the
        // trigger by default, so no extra label wiring needed here.
      >
        <ul className={styles.popoverList}>
          {contributors.map((c) => (
            <li key={c.login}>
              <Link
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.popoverItem}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={c.avatarUrl}
                  alt=""
                  className={styles.popoverAvatar}
                  width={20}
                  height={20}
                  loading="lazy"
                />
                <span className={styles.popoverName}>{c.name}</span>
                {/* Pinned cofounders can have commits=0 on files they
                 * never touched — omit the badge instead of showing "0",
                 * which would look like a bug. */}
                {c.commits > 0 ? (
                  <span className={styles.popoverCommits}>{c.commits}</span>
                ) : null}
              </Link>
            </li>
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  );
};


export default ContributorsOverflow;
