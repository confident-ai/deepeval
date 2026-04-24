"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

export type NavLink = {
  text: string;
  url: string;
  activeBase?: string;
  match?: "nested-url" | "exact";
  icon?: ReactNode;
};

export interface NavLinksProps {
  items: NavLink[];
}

export const navLinksListClassName = "flex items-center gap-3";

export const navLinkClassName =
  "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[12px] text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground [&_svg]:size-3.5 [&_svg]:shrink-0";

export function isNavLinkActive(pathname: string, item: NavLink) {
  const matchUrl = item.activeBase ?? item.url;
  const mode = item.match ?? "nested-url";
  if (mode === "exact") return pathname === matchUrl;
  return pathname === matchUrl || pathname.startsWith(`${matchUrl}/`);
}

type NavLinkItemProps = {
  item: NavLink;
  pathname: string;
};

export const NavLinkItem: React.FC<NavLinkItemProps> = ({ item, pathname }) => {
  const active = isNavLinkActive(pathname, item);

  return (
    <li>
      <Link
        href={item.url}
        data-active={active}
        className={navLinkClassName}
      >
        {item.icon}
        {item.text}
      </Link>
    </li>
  );
};

const NavLinks: React.FC<NavLinksProps> = ({ items }) => {
  const pathname = usePathname();

  return (
    <ul className={navLinksListClassName}>
      {items.map((item) => (
        <NavLinkItem key={item.url} item={item} pathname={pathname} />
      ))}
    </ul>
  );
};


export default NavLinks;
