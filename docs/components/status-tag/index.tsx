import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import styles from "./status-tag.module.scss";

type StatusTagProps = {
  children: ReactNode;
  className?: string;
  /** `action` matches the Copy Markdown / Open button height on docs pages. */
  size?: "compact" | "action";
};

/** Uppercase amber status chip (same surface as the TypeScript "BETA" tag). */
export function StatusTag({
  children,
  className,
  size = "compact",
}: StatusTagProps) {
  return (
    <span
      className={cn(
        styles.tag,
        size === "action" && styles.tagAction,
        className,
      )}
    >
      {children}
    </span>
  );
}
