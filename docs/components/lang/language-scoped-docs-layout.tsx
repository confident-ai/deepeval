'use client';

import { useMemo } from 'react';
import { DocsLayout, type DocsLayoutProps } from 'fumadocs-ui/layouts/notebook';
import { useLanguage } from '@/components/lang/language-provider';
import { filterPageTreeByLanguage } from '@/lib/lang/page-tree';

/**
 * `DocsLayout` with the page tree pruned to the reader's language, which also
 * takes the prev/next footer along with it.
 */
export const LanguageScopedDocsLayout = ({
  tree,
  ...props
}: DocsLayoutProps) => {
  const { language } = useLanguage();
  const scopedTree = useMemo(
    () => filterPageTreeByLanguage(tree, language),
    [tree, language],
  );

  return <DocsLayout tree={scopedTree} {...props} />;
};
