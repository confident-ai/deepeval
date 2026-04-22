"use client";

/**
 * Top-nav "Proudly Open-Source" button — a link to the deepeval
 * GitHub repo that surfaces the live star count next to the label.
 *
 * Why client-side fetch:
 *   `NavHeader` is already a Client Component (`useNotebookLayout`),
 *   and wiring a Server Component through Fumadocs' `slots.header`
 *   requires passing a wrapper that closes over pre-fetched data —
 *   doable but noisy for a single number. GitHub's unauthenticated
 *   REST endpoint allows ~60 req/hr per IP, and we cache the result
 *   in localStorage for 1h, so the typical visitor triggers at most
 *   one request per browser session.
 *
 * UX notes:
 *   - First paint uses whatever's in localStorage (if any), so
 *     returning visitors see a count immediately.
 *   - If no cache or cache is stale, we fetch in the background
 *     and update in place; no loading spinner (the number would
 *     flicker, which is worse than a brief em-dash).
 *   - If the fetch fails (offline, rate-limited, GitHub outage),
 *     the count stays as an em-dash; the link itself still works.
 */
import { useEffect, useState } from "react";
import { Star } from "lucide-react";
import { gitConfig } from "@/lib/shared";
import styles from "./GithubStarButton.module.scss";

/**
 * GitHub Octocat mark, inlined.
 *
 * Why not `lucide-react`'s `Github`: lucide removed all brand icons
 * (Github, Twitter, Discord, …) in v0.475+ because brand marks are
 * trademarks and shouldn't live in a generic icon set. We inline the
 * canonical Octocat path (https://github.com/logos, MIT-licensable
 * for attribution use) so we don't take a second icon-library
 * dependency just for one mark.
 *
 * `currentColor` on `fill` lets Tailwind's `text-*` utilities paint it
 * — same ergonomic as a lucide icon. `size-3.5` on the parent still
 * controls its dimensions because we don't hardcode width/height.
 */
function GithubMark(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
      {...props}
    >
      <path d="M12 .5C5.73.5.5 5.74.5 12.02c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.52-1.34-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.18 1.18.92-.26 1.9-.39 2.88-.39s1.96.13 2.88.39c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.77.12 3.06.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.4-5.25 5.68.41.36.78 1.07.78 2.16 0 1.56-.02 2.82-.02 3.21 0 .31.21.67.8.55C20.71 21.4 24 17.09 24 12.02 24 5.74 18.77.5 12 .5z" />
    </svg>
  );
}

type Props = {
  className?: string;
};

const CACHE_KEY = `gh-stars:${gitConfig.user}/${gitConfig.repo}`;
// 1 hour — long enough to keep traffic off GitHub's public rate-limit
// and short enough that a newly-starred repo shows the updated count
// for returning visitors the same day.
const CACHE_TTL_MS = 60 * 60 * 1000;

type Cached = { count: number; ts: number };

function readCache(): number | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Cached;
    if (
      typeof parsed?.count !== "number" ||
      typeof parsed?.ts !== "number" ||
      Date.now() - parsed.ts > CACHE_TTL_MS
    ) {
      return null;
    }
    return parsed.count;
  } catch {
    return null;
  }
}

function writeCache(count: number) {
  try {
    window.localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ count, ts: Date.now() } satisfies Cached),
    );
  } catch {
    // localStorage can throw in private modes / quota-exceeded —
    // cache is a nice-to-have, never block the UI on it.
  }
}

// Compact number formatter: 1234 → "1.2k", 12_345 → "12k", <1k → raw.
// Avoids `Intl.NumberFormat({ notation: "compact" })` because it
// returns "1.2K" (capital) and locale-dependent strings; we want a
// stable, terse look that matches the rest of the blueprint UI.
function formatCount(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1000;
  return (k < 10 ? k.toFixed(1).replace(/\.0$/, "") : Math.round(k).toString()) + "k";
}

export default function GithubStarButton({ className }: Props) {
  // Seed state from cache synchronously so the first paint already
  // shows a number for returning visitors. `useState` initializer
  // runs once per mount; localStorage reads are cheap.
  const [count, setCount] = useState<number | null>(() => readCache());

  useEffect(() => {
    // If cache was fresh, we already have a count — skip the fetch.
    // `readCache` returns `null` when the cache is expired, so this
    // condition only short-circuits when we trust what we have.
    if (readCache() !== null) return;

    let cancelled = false;
    fetch(`https://api.github.com/repos/${gitConfig.user}/${gitConfig.repo}`, {
      headers: { Accept: "application/vnd.github+json" },
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((j: { stargazers_count?: unknown }) => {
        if (cancelled) return;
        const n = j.stargazers_count;
        if (typeof n === "number" && Number.isFinite(n)) {
          setCount(n);
          writeCache(n);
        }
      })
      .catch(() => {
        // Silent: if GitHub's rate-limited or unreachable, we keep
        // whatever count we had (possibly null → em-dash). No toast,
        // no console noise — this is a vanity counter, not critical.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const href = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;
  const countLabel = count !== null ? formatCount(count) : "—";

  // All visuals (layout, solid border, the animated shine ring on
  // `::after`, responsive label/separator visibility, amber star
  // tint) live in `GithubStarButton.module.scss`. The component
  // itself only handles data + semantics.
  //
  // `data-active="false"` opts this button into the blueprint
  // callout's hover rule in `app/global.css`
  // (`#nd-subnav :where(a, button)[data-active]:hover`), so the
  // diagonal hatch fill + four corner marks fire on hover. The
  // shine ring stays on `::after` and the callout corner marks
  // live on `::before`, so the two treatments never fight for
  // the same pseudo-element.
  //
  // The parent-supplied `className` is expected to be a Tailwind
  // string for orthogonal concerns (e.g. `max-md:hidden` for
  // viewport-level visibility); it appends after `styles.root`
  // so ordering-sensitive utilities still win over the module
  // rules.
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-active="false"
      aria-label={
        count !== null
          ? `${gitConfig.repo} on GitHub — ${count.toLocaleString()} stars`
          : `${gitConfig.repo} on GitHub`
      }
      className={`${styles.root}${className ? ` ${className}` : ""}`}
    >
      <GithubMark />
      <span className={styles.label}>Proudly OS</span>
      <span className={styles.count}>
        <Star className={styles.star} />
        <span className={styles.countLabel}>{countLabel}</span>
      </span>
    </a>
  );
}
