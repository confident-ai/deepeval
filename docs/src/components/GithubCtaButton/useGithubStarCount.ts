"use client";

import { useEffect, useState } from "react";
import { gitConfig } from "@/lib/shared";

const CACHE_KEY = `gh-stars:${gitConfig.user}/${gitConfig.repo}`;
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
  } catch {}
}

export function useGithubStarCount(): number | null {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    const cached = readCache();
    if (cached !== null) {
      setCount(cached);
      return;
    }

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
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  return count;
}

export function formatStarCount(n: number): string {
  if (n < 1000) return String(n);
  const k = n / 1000;
  return `${k.toFixed(1)}k`;
}
