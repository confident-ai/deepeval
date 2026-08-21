import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { StatusTag } from "@/components/status-tag";
import styles from "@/components/status-tag/status-tag.module.scss";

type BetaMarkProps = {
  className?: string;
};

/**
 * Compact β marker for sidebar labels. Same amber as the TypeScript BETA
 * chip — greek letter only, no border/background, to save sidebar width.
 */
export function BetaMark({ className }: BetaMarkProps) {
  return (
    <span className={cn(styles.mark, className)} title="Beta" aria-label="Beta">
      β
    </span>
  );
}

/** Full "BETA" chip for the page header (left of Copy Markdown). */
export function BetaBadge({ className }: BetaMarkProps) {
  return (
    <StatusTag size="action" className={className}>
      Beta
    </StatusTag>
  );
}

/** Append a β mark to a page-tree node name when the page is beta. */
export function withBetaMark(name: ReactNode): ReactNode {
  return (
    <>
      {name}
      <BetaMark />
    </>
  );
}
