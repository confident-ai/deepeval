import type * as PageTree from 'fumadocs-core/page-tree';
import { LANGUAGE_IDS, type Language } from './languages';
import type { WithLanguages } from './page-tree';

/**
 * No page may support a language its folder's index page doesn't: the sidebar
 * filter drops such a folder wholesale, stranding the child with nothing left
 * linking to it. Throws so `next build` fails and `next dev` reports it on the
 * first request.
 */
export function assertPageTreeLanguages(
  sources: Record<string, { getPageTree: () => PageTree.Root }>,
) {
  const problems: string[] = [];

  for (const [section, source] of Object.entries(sources)) {
    visit(source.getPageTree().children, undefined, section, problems);
  }

  if (problems.length === 0) return;

  throw new Error(
    [
      `Unreachable pages in the language-filtered sidebar (${problems.length}):`,
      ...problems.map((problem) => `  - ${problem}`),
      '',
      "Either add the missing language to the parent's `languages` frontmatter,",
      "or drop it from the child's.",
    ].join('\n'),
  );
}

/** The nearest enclosing index page, which caps what its subtree may declare. */
type Ceiling = { url: string; languages: Language[] };

function visit(
  nodes: PageTree.Node[],
  ceiling: Ceiling | undefined,
  section: string,
  problems: string[],
) {
  for (const node of nodes) {
    if (node.type === 'separator') continue;

    if (node.type === 'page') {
      check(node, ceiling, section, problems);
      continue;
    }

    // Route-group folders have no index page and so cap nothing.
    const index = node.index as WithLanguages<PageTree.Item> | undefined;
    if (index) check(index, ceiling, section, problems);

    visit(
      node.children,
      index?.languages
        ? { url: index.url, languages: index.languages }
        : ceiling,
      section,
      problems,
    );
  }
}

function check(
  node: WithLanguages<PageTree.Item>,
  ceiling: Ceiling | undefined,
  section: string,
  problems: string[],
) {
  if (!ceiling || node.url === ceiling.url) return;

  // An undeclared page renders for everyone, so it out-reaches a narrowed
  // parent just the same.
  const declared = node.languages ?? LANGUAGE_IDS;
  const excess = declared.filter((lang) => !ceiling.languages.includes(lang));
  if (excess.length === 0) return;

  problems.push(
    `[${section}] ${node.url} supports ${excess.join(', ')} but its parent ` +
      `${ceiling.url} does not, so it would be hidden with the parent.`,
  );
}
