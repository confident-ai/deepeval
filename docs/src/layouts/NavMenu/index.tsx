"use client";

import * as Popover from "@radix-ui/react-popover";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu } from "lucide-react";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import { twMerge } from "tailwind-merge";
import { isNavLinkActive, type NavLink } from "@/src/layouts/NavLinks";
import styles from "./NavMenu.module.scss";

export interface NavMenuProps {
  items: NavLink[];
}

const NavMenu: React.FC<NavMenuProps> = ({ items }) => {
  const pathname = usePathname();

  return (
    <Popover.Root>
      <Popover.Trigger
        aria-label="Open navigation menu"
        className={twMerge(
          buttonVariants({ size: "icon-sm", color: "secondary" }),
          "text-fd-muted-foreground rounded-none",
          styles.trigger,
        )}
      >
        <Menu />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          collisionPadding={8}
          className={styles.content}
        >
          <ul className={styles.list}>
            {items.map((item) => {
              const active = isNavLinkActive(pathname, item);
              return (
                <li key={item.url}>
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
};


export default NavMenu;
