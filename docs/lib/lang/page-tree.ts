import type * as PageTree from 'fumadocs-core/page-tree';
import type { Language } from './languages';

/** Attached by the page-tree transformer in `lib/source.ts`. */
export type WithLanguages<T> = T & { languages?: Language[] };

function supportsLanguage(node: PageTree.Item, language: Language) {
  const { languages } = node as WithLanguages<PageTree.Item>;
  return !languages || languages.includes(language);
}

export function filterPageTreeByLanguage(
  tree: PageTree.Root,
  language: Language,
): PageTree.Root {
  return {
    ...tree,
    // Fumadocs memoizes the tree on `$id` alone, so a copy reusing the original
    // id is discarded and the unfiltered tree keeps rendering.
    $id: `${tree.$id ?? 'root'}:${language}`,
    children: filterNodes(tree.children, language),
  };
}

function filterNodes(
  nodes: PageTree.Node[],
  language: Language,
): PageTree.Node[] {
  const kept: PageTree.Node[] = [];

  for (const node of nodes) {
    if (node.type === 'folder') {
      const folder = filterFolder(node, language);
      if (folder) kept.push(folder);
    } else if (node.type === 'separator' || supportsLanguage(node, language)) {
      kept.push(node);
    }
  }

  return dropEmptyGroups(kept);
}

function filterFolder(
  folder: PageTree.Folder,
  language: Language,
): PageTree.Folder | undefined {
  // Safe to drop wholesale: `lib/lang/validate.ts` guarantees no child supports
  // a language the index page doesn't.
  if (folder.index && !supportsLanguage(folder.index, language)) return;

  const children = filterNodes(folder.children, language);
  if (children.length === 0 && !folder.index) return;

  return { ...folder, children };
}

/** A separator labels the run of siblings beneath it, so it goes when that run does. */
function dropEmptyGroups(nodes: PageTree.Node[]): PageTree.Node[] {
  const kept: PageTree.Node[] = [];
  let groupHasContent = false;

  for (let i = nodes.length - 1; i >= 0; i--) {
    const node = nodes[i];

    if (node.type !== 'separator') {
      groupHasContent = true;
    } else if (groupHasContent) {
      groupHasContent = false;
    } else {
      continue;
    }

    kept.push(node);
  }

  return kept.reverse();
}
