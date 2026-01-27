import React from 'react';
import clsx from 'clsx';
import TOCItems from '@theme/TOCItems';
import styles from './styles.module.scss';

// Using a custom className
// This prevents TOCInline/TOCCollapsible getting highlighted by mistake
const LINK_CLASS_NAME = 'table-of-contents__link toc-highlight';
const LINK_ACTIVE_CLASS_NAME = 'table-of-contents__link--active';

export default function TOC({ className, ...props }) {
  return (
    <div className={clsx(styles.tableOfContents, className)}>
      {/* Scrollable container for TOC items */}
      <div className={clsx(styles.tocItemsContainer, 'thin-scrollbar')}>
        <TOCItems
          {...props}
          linkClassName={LINK_CLASS_NAME}
          linkActiveClassName={LINK_ACTIVE_CLASS_NAME}
        />
      </div>
      {/* PromoCard always visible at the bottom */}
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
            (window.location.href = 'https://app.confident-ai.com')
          }
        >
          Try it for Free
        </div>
      </div>
    </div>
  );
}
