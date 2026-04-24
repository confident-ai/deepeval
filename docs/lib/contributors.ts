/**
 * Typed view of the build-time contributors manifest (see
 * `scripts/generate-contributors.mjs`). Keyed by repo-relative file
 * path like `content/docs/getting-started.mdx`.
 *
 * The JSON is statically imported so bundling picks it up at build
 * time without a runtime fetch. An empty `{}` (default / no-git-repo
 * state) is valid — every lookup just returns an empty list and the
 * UI renders nothing.
 */
import manifest from "./generated/contributors.json";

export type Contributor = {
  readonly login: string;
  readonly name: string;
  readonly avatarUrl: string;
  readonly url: string;
  readonly commits: number;
};

type Manifest = Record<string, Contributor[]>;

const typedManifest = manifest as Manifest;

/**
 * Look up contributors for a page given its section `contentDir`
 * (e.g. `content/docs`) and the loader's `page.path`. These are the
 * same two inputs used to build the "Edit on GitHub" URL, which keeps
 * the manifest-key scheme trivial to reason about.
 */
export function getPageContributors(
  contentDir: string,
  pagePath: string,
): Contributor[] {
  return typedManifest[`${contentDir}/${pagePath}`] ?? [];
}
