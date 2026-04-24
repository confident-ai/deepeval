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
import { twMerge } from "tailwind-merge";
import SiteTopNav from "@/src/layouts/SiteTopNav";

// Fumadocs doesn't ship `cn` (from `utils/cn`) or `LinkItem` (from
// `layouts/shared/client`) as public exports — both live inside
// internal paths. We inline the two bits of functionality we need:
// `twMerge` handles Tailwind class conflict resolution (the only
// reason `cn` exists upstream), and icon nav items are plain links
// with a known shape (`{ type: "icon", url, icon, label, external }`)
// so a native `<a>` is enough.

const NavHeader: React.FC<React.ComponentProps<"header">> = (props: React.ComponentProps<"header">) => {
  const {
    slots,
    navItems,
    isNavTransparent,
    props: { sidebar },
  } = useNotebookLayout();
  const { open } = slots.sidebar?.useSidebar?.() ?? {};
  const sidebarCollapsible = sidebar.collapsible ?? true;

  void navItems;

  return (
    <SiteTopNav
      variant="docs"
      dataTransparent={isNavTransparent && !open}
      navTitle={slots.navTitle ?? false}
      themeSwitch={slots.themeSwitch ?? false}
      collapseTrigger={
        sidebarCollapsible && slots.sidebar
          ? slots.sidebar.collapseTrigger
          : false
      }
      headerProps={{
        ...props,
        className: twMerge(props.className),
      }}
    />
  );
};


export default NavHeader;
