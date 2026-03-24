import React, {type ReactNode} from 'react';
import {useWindowSize} from '@docusaurus/theme-common';
import DocSidebarDesktop from '@theme/DocSidebar/Desktop';
import DocSidebarMobile from '@theme/DocSidebar/Mobile';
import type {Props} from '@theme/DocSidebar';
import SearchBar from '@theme/SearchBar';
import styles from './styles.module.scss';

export default function DocSidebar(props: Props): ReactNode {
  const windowSize = useWindowSize();

  // Desktop sidebar visible on hydration: need SSR rendering
  const shouldRenderSidebarDesktop =
    windowSize === 'desktop' || windowSize === 'ssr';

  // Mobile sidebar not visible on hydration: can avoid SSR rendering
  const shouldRenderSidebarMobile = windowSize === 'mobile';

  return (
    <>
      {shouldRenderSidebarDesktop && (
        <div className={styles.sidebarContainer}>
          <div className={styles.searchBarContainer}>
            <SearchBar />
          </div>
          <DocSidebarDesktop {...props} />
        </div>
      )}
      {shouldRenderSidebarMobile && <DocSidebarMobile {...props} />}
    </>
  );
}