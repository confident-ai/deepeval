'use client';

import React, { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { htmlToMarkdown } from "@site/src/utils/html-to-markdown";
import styles from "./CopyPageButton.module.scss";

function useDoc() {
  return { metadata: { title: "" } };
}

export default function CopyPageButton(): React.ReactNode {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [portalTarget, setPortalTarget] = useState<Element | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { metadata } = useDoc();

  useEffect(() => {
    const mdContainer = document.querySelector(".theme-doc-markdown");
    if (!mdContainer) return;
    const header = mdContainer.querySelector(":scope > header");
    if (header) {
      setPortalTarget(header);
    }
  }, []);

  useEffect(() => {
    const onClickOutside = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const getPageMarkdown = useCallback(() => {
    const contentEl = document.querySelector(".theme-doc-markdown");
    if (!contentEl) return "";
    const md = htmlToMarkdown(contentEl);
    return `# ${metadata.title}\n\nSource: ${window.location.href}\n\n${md}`;
  }, [metadata.title]);

  const copyAndDo = useCallback(
    async (afterCopy?: () => void) => {
      const md = getPageMarkdown();
      await navigator.clipboard.writeText(md);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      setIsOpen(false);
      afterCopy?.();
    },
    [getPageMarkdown]
  );

  const handleViewMarkdown = useCallback(() => {
    const md = getPageMarkdown();
    const blob = new Blob([md], { type: "text/plain;charset=utf-8" });
    window.open(URL.createObjectURL(blob), "_blank");
    setIsOpen(false);
  }, [getPageMarkdown]);

  if (!portalTarget) return null;

  return createPortal(
    <div className={styles.container} ref={containerRef} data-copy-page-ignore>
      <button
        className={styles.trigger}
        onClick={() => setIsOpen((o) => !o)}
        aria-expanded={isOpen}
        aria-label="Copy page options"
      >
        <CopyIcon size={14} />
        <span>{copied ? "Copied!" : "Copy page"}</span>
        <svg
          className={`${styles.chevron} ${isOpen ? styles.chevronOpen : ""}`}
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {isOpen && (
        <div className={styles.dropdown}>
          <button className={styles.dropdownItem} onClick={() => copyAndDo()}>
            <CopyIcon size={16} />
            <div className={styles.itemText}>
              <div className={styles.itemLabel}>Copy page</div>
              <div className={styles.itemDescription}>
                Copy this page as Markdown for LLMs
              </div>
            </div>
          </button>

          <button className={styles.dropdownItem} onClick={handleViewMarkdown}>
            <FileTextIcon />
            <div className={styles.itemText}>
              <div className={styles.itemLabel}>View as Markdown</div>
              <div className={styles.itemDescription}>
                View this page as plain text
              </div>
            </div>
            <ExternalLinkIcon />
          </button>

          <button
            className={styles.dropdownItem}
            onClick={() => {
              const q = encodeURIComponent(
                `Read ${window.location.href} so I can ask questions about it`
              );
              window.open(`https://claude.ai/new?q=${q}`, "_blank");
              setIsOpen(false);
            }}
          >
            <span className={styles.aiIcon}>AI</span>
            <div className={styles.itemText}>
              <div className={styles.itemLabel}>Open in Claude</div>
              <div className={styles.itemDescription}>
                Ask questions about this page
              </div>
            </div>
            <ExternalLinkIcon />
          </button>

          <button
            className={styles.dropdownItem}
            onClick={() => {
              const q = encodeURIComponent(
                `Read ${window.location.href} so I can ask questions about it`
              );
              window.open(
                `https://chat.openai.com/?hint=search&q=${q}`,
                "_blank"
              );
              setIsOpen(false);
            }}
          >
            <SparklesIcon />
            <div className={styles.itemText}>
              <div className={styles.itemLabel}>Open in ChatGPT</div>
              <div className={styles.itemDescription}>
                Ask questions about this page
              </div>
            </div>
            <ExternalLinkIcon />
          </button>
        </div>
      )}
    </div>,
    portalTarget
  );
}

function CopyIcon({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  );
}

function FileTextIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <line x1="10" y1="9" x2="8" y2="9" />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg
      className={styles.externalIcon}
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}
