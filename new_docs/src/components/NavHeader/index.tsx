"use client";

/**
 * Custom notebook header that replaces Fumadocs' default `Header` slot.
 *
 * Why: Fumadocs' stock header body is a plain flex row with fixed
 * `px-4 md:px-6` padding, so its three visual sections don't line up
 * with the three columns of the outer docs grid (sidebar / main / toc).
 * We want hairline-aligned edges. Easiest way: mirror the outer grid's
 * column template here.
 *
 * How: the outer grid (see `node_modules/fumadocs-ui/dist/layouts/
 * notebook/slots/container.js`) uses
 *     var(--fd-sidebar-col)  minmax(0, 1fr)  var(--fd-toc-width)
 * for the three docs columns (ignoring the outer gutters). We
 * reproduce that exact template on our `data-header-body` so the
 * header cells land on the same vertical lines.
 *
 * Everything else is taken faithfully from Fumadocs' default Header so
 * we keep: sticky offset, `data-transparent` backdrop flip, nav-mode
 * semantics, button variants, collapse trigger wiring, and the mobile
 * search / hamburger branch. Reference: `layouts/notebook/slots/
 * header.js` in fumadocs-ui@16.
 */

import { useNotebookLayout } from "fumadocs-ui/layouts/notebook";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import { twMerge } from "tailwind-merge";
import { Sidebar } from "lucide-react";
import NavLinks from "@/src/components/NavLinks";
import NavMenu from "@/src/components/NavMenu";
import GithubStarButton from "@/src/components/GithubStarButton";
import { navLinks } from "@/lib/layout.shared";
import styles from "./NavHeader.module.scss";

// Fumadocs doesn't ship `cn` (from `utils/cn`) or `LinkItem` (from
// `layouts/shared/client`) as public exports — both live inside
// internal paths. We inline the two bits of functionality we need:
// `twMerge` handles Tailwind class conflict resolution (the only
// reason `cn` exists upstream), and icon nav items are plain links
// with a known shape (`{ type: "icon", url, icon, label, external }`)
// so a native `<a>` is enough.

export default function NavHeader(
  props: React.ComponentProps<"header">,
) {
  const {
    slots,
    navItems,
    isNavTransparent,
    props: { sidebar },
  } = useNotebookLayout();
  const { open } = slots.sidebar?.useSidebar?.() ?? {};
  const sidebarCollapsible = sidebar.collapsible ?? true;

  // `navItems` previously sourced the GitHub icon (via `githubUrl` in
  // baseOptions). We now render our own <GithubStarButton /> with a
  // live star count + label, so the built-in icon entry is ignored.
  // Keeping `navItems` on the hook call because Fumadocs may emit
  // other icon types in the future and we want to surface them here
  // without a second plumbing round-trip.
  void navItems;

  return (
    <header
      id="nd-subnav"
      data-transparent={isNavTransparent && !open}
      {...props}
      className={twMerge(
        // Copied verbatim from Fumadocs' default notebook header so
        // sticky offsets, transparency behavior, and header-height
        // token all continue to work. The only thing we strip is the
        // `flex flex-col` — our inner body is a CSS grid instead.
        "sticky [grid-area:header] top-(--fd-docs-row-1) z-10 backdrop-blur-sm transition-colors data-[transparent=false]:bg-fd-background/80 layout:[--fd-header-height:--spacing(14)]",
        props.className,
      )}
    >
      <div data-header-body="" className={styles.body}>
        <div className={styles.logoCell}>
          {slots.navTitle ? (
            <slots.navTitle className={styles.logo} />
          ) : null}
        </div>

        <div className={styles.mainCell}>
          <div className={styles.mainNavLinks}>
            <NavLinks items={navLinks} />
          </div>
          <div className={styles.mainMenuTrigger}>
            <NavMenu items={navLinks} />
          </div>
          <div className={styles.mainGithub}>
            <GithubStarButton />
          </div>
        </div>

        <div className={styles.utilsCell}>
          <div className={styles.utilsDesktop}>
            <div className={styles.utilsGithub}>
              <GithubStarButton />
            </div>
            {slots.themeSwitch ? (
              <slots.themeSwitch className="rounded-none" />
            ) : null}
            {sidebarCollapsible && slots.sidebar ? (
              <slots.sidebar.collapseTrigger
                className={twMerge(
                  buttonVariants({ size: "icon-sm", color: "secondary" }),
                  "text-fd-muted-foreground rounded-none",
                )}
              >
                <Sidebar />
              </slots.sidebar.collapseTrigger>
            ) : null}
          </div>

          <div className={styles.utilsMobile}>
            <NavMenu items={navLinks} />
          </div>
        </div>
      </div>
    </header>
  );
}
