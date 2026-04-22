"use client";

// Left-aligned nav strip rendered next to the logo. We don't use
// Fumadocs' `links` option because that places the text items on the
// far right of the header; we want the classic "Logo | Nav — Search |
// Icons" layout (Tailwind / Next.js docs style) where nav items hug
// the logo. Passing this component via `nav.children` drops it into
// the same flex cell as `navTitle`.
//
// Styling + active-state logic mirrors `NavbarLinkItem` inside
// `fumadocs-ui/layouts/notebook/slots/header.js` so the items look
// identical to what Fumadocs would have rendered on the right.

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

type NavLink = {
  text: string;
  url: string;
  activeBase?: string;
  // Fumadocs' own `active: 'nested-url'` semantics: match any path that
  // starts with `url`. All our links use that mode.
  match?: "nested-url" | "exact";
  // Optional leading icon — rendered at 16px to match the link text
  // baseline. Passed as a ReactNode so callers can inline whatever
  // Lucide (or inline SVG) they like without this component having
  // to know about the icon library.
  icon?: ReactNode;
};

export interface NavLinksProps {
  items: NavLink[];
  className?: string;
}

function isActive(pathname: string, item: NavLink) {
  const matchUrl = item.activeBase ?? item.url;
  const mode = item.match ?? "nested-url";
  if (mode === "exact") return pathname === matchUrl;
  return pathname === matchUrl || pathname.startsWith(`${matchUrl}/`);
}

export default function NavLinks({ items, className }: NavLinksProps) {
  const pathname = usePathname();
  return (
    <ul
      className={
        // `gap-3` (12px) gives adjacent items' corner marks enough
        // clearance: the blueprint callout extends
        // `--fd-callout-offset` (2px) outside each link, so two
        // neighbours together eat 4px of the gap just with their
        // outward-facing L brackets. 12px gap → 8px visible breathing
        // room between corner marks, which matches the rhythm of the
        // header's other elements without feeling airy.
        "flex items-center gap-3" +
        (className ? " " + className : "")
      }
    >
      {items.map((item) => {
        const active = isActive(pathname, item);
        return (
          <li key={item.url}>
            <Link
              href={item.url}
              data-active={active}
              // Compact nav button: 12px label + 14px icon, with a
              // slightly tighter vertical pad so the row breathes a
              // bit more in the constrained header band.
              // `[&_svg]:size-3.5` normalises every Lucide icon to
              // 14px regardless of its intrinsic viewBox.
              //
              // `px-2 py-1 rounded-md` is applied to *every* link,
              // not just the active one, so the row doesn't reflow
              // when a new page becomes active — the active state
              // (see `app/global.css`) just fills that same padded
              // box with the diagonal-hatch callout.
              className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[12px] text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground [&_svg]:size-3.5 [&_svg]:shrink-0"
            >
              {item.icon}
              {item.text}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
