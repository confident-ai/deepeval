"use client";

/**
 * Search trigger rendered at the top of the sidebar (via
 * `sidebar.banner`). We render our own button instead of Fumadocs'
 * stock `searchTrigger.full` so the sidebar stays free of the
 * built-in Cmd/Ctrl+K shortcut badges.
 *
 * Same trigger is shown in both the desktop sticky aside and the
 * mobile sidebar drawer, so search is reachable in either surface.
 */

import { Search } from "lucide-react";
import { useI18n } from "fumadocs-ui/contexts/i18n";
import { useSearchContext } from "fumadocs-ui/contexts/search";

const SidebarSearch: React.FC = () => {
  const { enabled, setOpenSearch } = useSearchContext();
  const { text } = useI18n();
  if (!enabled) return null;

  // `w-full` fills the 268px sidebar column (minus the banner
  // wrapper's p-4). `rounded-xl` + `ps-2.5` mirror Fumadocs' own
  // navMode="top" trigger styling for visual consistency.
  return (
    <button
      type="button"
      data-search-full=""
      className="inline-flex w-full items-center gap-2 rounded-xl border bg-fd-secondary/50 p-1.5 ps-2.5 text-sm text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-accent-foreground"
      aria-label="Open Search"
      onClick={() => setOpenSearch(true)}
    >
      <Search className="size-4" />
      {text.search}
    </button>
  );
};


export default SidebarSearch;
