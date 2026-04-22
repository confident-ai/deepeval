"use client";

/**
 * Hamburger dropdown for the top nav — the mobile / sub-lg counterpart
 * to `NavLinks`. Below the `lg` breakpoint (1024px) the inline link
 * strip in `NavLinks` hides itself (`max-lg:hidden`); this component
 * shows in its place so the section switcher (Docs / Guides /
 * Tutorials / Integrations / Changelog / Blog) stays reachable.
 *
 * Trigger is a square 28px icon-button styled like the other utility
 * chrome in col 3 (theme switch, sidebar collapse) — same
 * `buttonVariants` call, same `rounded-none` so it participates in the
 * global `data-header-body` flatten-border-radius rule. Content is a
 * Radix Popover portal: renders OUTSIDE `#nd-subnav`, so the
 * blueprint-callout selector (`#nd-subnav :where(a, button)[data-active]…`)
 * does NOT apply. That's intentional — a dropdown surface wants a
 * simple colored-hover indicator, not the corner-mark L shapes that
 * are calibrated for the flat header strip.
 *
 * Visibility is controlled in CSS (`.trigger` is `display: none` at
 * `min-width: 1024px`) rather than in JSX, so this component is
 * always mounted — which means `@radix-ui/react-popover` keeps its
 * internal state consistent across breakpoint flips (e.g. user opens
 * menu, rotates tablet into desktop orientation, dropdown stays
 * closed on the way back down).
 */

import * as Popover from "@radix-ui/react-popover";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu } from "lucide-react";
import type { ReactNode } from "react";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import { twMerge } from "tailwind-merge";
import styles from "./NavMenu.module.scss";

type NavLink = {
  text: string;
  url: string;
  match?: "nested-url" | "exact";
  icon?: ReactNode;
};

export interface NavMenuProps {
  items: NavLink[];
  className?: string;
}

function isActive(pathname: string, item: NavLink) {
  const mode = item.match ?? "nested-url";
  if (mode === "exact") return pathname === item.url;
  return pathname === item.url || pathname.startsWith(`${item.url}/`);
}

export default function NavMenu({ items, className }: NavMenuProps) {
  const pathname = usePathname();

  return (
    <Popover.Root>
      <Popover.Trigger
        aria-label="Open navigation menu"
        className={twMerge(
          buttonVariants({ size: "icon-sm", color: "secondary" }),
          // `text-fd-muted-foreground` + `rounded-none` match the
          // col-3 icon buttons exactly so the hamburger feels like
          // part of the same utility family.
          "text-fd-muted-foreground rounded-none",
          styles.trigger,
          className,
        )}
      >
        <Menu />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          /* Right-anchored to match the trigger's col 3 position:
           * we want the panel to extend leftward from the button's
           * right edge, not rightward off the viewport. */
          align="end"
          sideOffset={8}
          collisionPadding={8}
          className={styles.content}
        >
          <ul className={styles.list}>
            {items.map((item) => {
              const active = isActive(pathname, item);
              return (
                <li key={item.url}>
                  {/* `Popover.Close asChild` wraps each link so a
                   *  click dismisses the menu — Radix auto-closes
                   *  on outside click, but not on inside click of
                   *  an `<a>` that navigates client-side. Without
                   *  this wrap the menu stays open across route
                   *  changes, which looks broken. */}
                  <Popover.Close asChild>
                    <Link
                      href={item.url}
                      data-active={active}
                      className={styles.item}
                    >
                      <span className={styles.icon}>{item.icon}</span>
                      <span>{item.text}</span>
                    </Link>
                  </Popover.Close>
                </li>
              );
            })}
          </ul>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
