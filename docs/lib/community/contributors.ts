import contributors from "@/lib/generated/repo-contributors.json";

export type RepoContributor = {
  login: string;
  avatarUrl: string;
  url: string;
  contributions: number;
};

// Regenerated on every build by `npm run repo-contributors`.
const list = (contributors as RepoContributor[]).filter(
  (c) => !c.login.endsWith("[bot]"),
);

export const contributorCount = list.length;
export const contributionCount = list.reduce((sum, c) => sum + c.contributions, 0);

/** The top contributors by contributions, most first. */
export const topContributors = (n: number) =>
  [...list].sort((a, b) => b.contributions - a.contributions).slice(0, n);
