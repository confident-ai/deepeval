"use client";

import { useState, type ReactNode } from "react";
import { X } from "lucide-react";

// Same markup, storage key and CSS contract as the fumadocs <Banner>, minus
// its inline <script>. React 19 logs "Encountered a script tag while
// rendering React component" for that script on every client render. The
// dismissed check runs in <head> instead: add each bar id to
// lib/banner-restore.ts.
export function AnnouncementBar({
  id,
  height,
  changeLayout = true,
  children,
}: {
  id: string;
  height: string;
  changeLayout?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(true);
  const key = `nd-banner-${id}`;
  if (!open) return null;

  return (
    <div
      id={id}
      className="sticky top-0 z-40 flex flex-row items-center justify-center px-4 text-center text-sm font-medium bg-fd-secondary"
      style={{ height }}
    >
      {changeLayout ? <style>{`:root:not(.${key}) { --fd-banner-height: ${height}; }`}</style> : null}
      <style>{`.${key} #${id} { display: none; }`}</style>
      {children}
      <button
        type="button"
        aria-label="Close Banner"
        className="absolute end-2 top-1/2 -translate-y-1/2 inline-flex items-center justify-center p-1.5 text-fd-muted-foreground/50 hover:text-fd-accent-foreground [&_svg]:size-4"
        onClick={() => {
          setOpen(false);
          document.documentElement.classList.add(key);
          try {
            localStorage.setItem(key, "true");
          } catch {}
        }}
      >
        <X />
      </button>
    </div>
  );
}
