"use client";

/**
 * Search trigger rendered at the top of the sidebar (via
 * `sidebar.banner`). We can't just drop `slots.searchTrigger.full`
 * directly into `layout.shared` because `slots` only exists inside
 * the notebook layout context — so this thin client component hops
 * into that context and renders the wired trigger.
 *
 * Same trigger is shown in both the desktop sticky aside and the
 * mobile sidebar drawer, so search is reachable in either surface.
 */

import { useNotebookLayout } from "fumadocs-ui/layouts/notebook";

export default function SidebarSearch() {
  const { slots } = useNotebookLayout();
  if (!slots.searchTrigger) return null;
  // `w-full` fills the 268px sidebar column (minus the banner
  // wrapper's p-4). `rounded-xl` + `ps-2.5` mirror Fumadocs' own
  // navMode="top" trigger styling for visual consistency.
  return (
    <slots.searchTrigger.full
      hideIfDisabled
      className="w-full ps-2.5 rounded-xl"
    />
  );
}
