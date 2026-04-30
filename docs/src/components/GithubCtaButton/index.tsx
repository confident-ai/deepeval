"use client";

import Link from "next/link";
import { Star } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { gitConfig } from "@/lib/shared";
import { formatStarCount, useGithubStarCount } from "./useGithubStarCount";
import styles from "./GithubCtaButton.module.scss";

/**
 * Inlined GitHub Octocat mark.
 */
export const GithubMark: React.FC<React.SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 .5C5.73.5.5 5.74.5 12.02c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.52-1.34-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.18 1.18.92-.26 1.9-.39 2.88-.39s1.96.13 2.88.39c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.77.12 3.06.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.4-5.25 5.68.41.36.78 1.07.78 2.16 0 1.56-.02 2.82-.02 3.21 0 .31.21.67.8.55C20.71 21.4 24 17.09 24 12.02 24 5.74 18.77.5 12 .5z" />
    </svg>
  );
};

type GithubCtaButtonProps = {
  layout?: "full" | "inline";
  tone?: "inverse" | "secondary";
  alwaysCallout?: boolean;
};

const GithubCtaButton: React.FC<GithubCtaButtonProps> = ({
  layout = "full",
  tone = "inverse",
  alwaysCallout = false,
}) => {
  const count = useGithubStarCount();
  const href = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;
  const countLabel = count !== null ? formatStarCount(count) : "—";

  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={twMerge(
        styles.root,
        alwaysCallout && "fd-blueprint-callout",
        alwaysCallout && styles.highlighted
      )}
      data-layout={layout}
      data-tone={tone}
      aria-label={
        count !== null
          ? `Find us on Github — ${count.toLocaleString()} stars`
          : "Find us on Github"
      }
      data-callout
      data-button
    >
      <span className={styles.content}>
        <GithubMark />
        <span>Find us on Github</span>
      </span>
      <span className={styles.count}>
        <Star className={styles.star} />
        <span>{countLabel}</span>
      </span>
    </Link>
  );
};


export default GithubCtaButton;
