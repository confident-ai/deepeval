import { type ReactNode } from 'react';
import clsx from 'clsx';
import TOCItems from '@theme/TOCItems';
import type { Props } from '@theme/TOC';

import styles from './styles.module.scss';
import {
  CONFIDENT_HOSTS_BY_NAME,
  appendDeepEvalAttribution,
} from '@site/src/utils/utm';

const LINK_CLASS_NAME = 'table-of-contents__link toc-highlight';
const LINK_ACTIVE_CLASS_NAME = 'table-of-contents__link--active';

export default function TOC({ className, ...props }: Props): ReactNode {
  return (
    <div className={clsx(styles.tableOfContents, className)}>
      <div className={clsx(styles.tocItemsContainer, 'thin-scrollbar')}>
        <TOCItems
          {...props}
          linkClassName={LINK_CLASS_NAME}
          linkActiveClassName={LINK_ACTIVE_CLASS_NAME}
        />
      </div>
      <div className={styles.promoCard}>
        <img src="/icons/logo.svg" alt="Confident AI" className={styles.logo} />
        <div className={styles.heading}>
          Try DeepEval on Confident AI for FREE
        </div>
        <div className={styles.description}>
          View and save evaluation results, curate datasets and manage
          annotations, monitor online performance, trace for AI observability,
          and auto-optimize prompts.
        </div>
        <div
          className={styles.button}
          onClick={() =>
            (window.location.href = appendDeepEvalAttribution(
              CONFIDENT_HOSTS_BY_NAME.APP,
              { content: 'sidebar_promo' }
            ))
          }
        >
          Try for free
        </div>
      </div>
    </div>
  );
}
