"use client";

import {
  Children,
  isValidElement,
  useMemo,
  type ComponentProps,
  type ReactNode,
} from "react";
import type { BreadcrumbOptions } from "fumadocs-core/breadcrumb";
import { getBreadcrumbItemsFromPath } from "fumadocs-core/breadcrumb";
import { useTreeContext, useTreePath } from "fumadocs-ui/contexts/tree";
import Link from "fumadocs-core/link";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";

type PlainBreadcrumbProps = BreadcrumbOptions & ComponentProps<"div">;

/**
 * Drop non-text children from a page-tree `name`. Beta pages wrap the
 * label with <BetaMark /> for the sidebar; breadcrumbs should stay plain.
 */
function plainLabel(name: ReactNode): ReactNode {
  if (name == null || typeof name === "boolean") return null;
  if (typeof name === "string" || typeof name === "number") return name;

  if (Array.isArray(name)) {
    const parts = name
      .map(plainLabel)
      .filter((part) => part != null && part !== "");
    if (parts.length === 0) return null;
    if (parts.length === 1) return parts[0];
    return parts;
  }

  if (isValidElement<{ children?: ReactNode }>(name)) {
    return plainLabel(Children.toArray(name.props.children));
  }

  return null;
}

/**
 * Same trail as Fumadocs' notebook Breadcrumb, but with β marks stripped
 * out of labels so the sidebar can keep them in `node.name`.
 */
export function PlainBreadcrumb({
  includeRoot,
  includeSeparator,
  includePage,
  ...props
}: PlainBreadcrumbProps) {
  const path = useTreePath();
  const { root } = useTreeContext();
  const items = useMemo(() => {
    return getBreadcrumbItemsFromPath(root, path, {
      includePage,
      includeSeparator,
      includeRoot,
    }).map((item) => ({
      ...item,
      name: plainLabel(item.name) ?? item.name,
    }));
  }, [includePage, includeRoot, includeSeparator, path, root]);

  if (items.length === 0) return null;

  return (
    <div
      {...props}
      className={cn(
        "flex items-center gap-1.5 text-sm text-fd-muted-foreground",
        props.className,
      )}
    >
      {items.map((item, i) => {
        const className = cn(
          "truncate",
          i === items.length - 1 && "text-fd-primary font-medium",
        );
        return (
          <span key={i} className="contents">
            {i !== 0 ? <ChevronRight className="size-3.5 shrink-0" /> : null}
            {item.url ? (
              <Link
                href={item.url}
                className={cn(className, "transition-opacity hover:opacity-80")}
              >
                {item.name}
              </Link>
            ) : (
              <span className={className}>{item.name}</span>
            )}
          </span>
        );
      })}
    </div>
  );
}
