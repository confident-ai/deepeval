"use client";

import { useEffect, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import { twMerge } from "tailwind-merge";

// Same markup, storage key and CSS contract as fumadocs' <Banner>, minus its
// inline <script>: React 19 errors when it renders a <script> on the client
// (client navigation, Fast Refresh). bannerPrepaintScript in <head> does that
// script's job instead, adding `nd-banner-<id>` to <html> before first paint.
type AnnouncementBarProps = {
  id: string;
  height: string;
  // Sets --fd-banner-height so the docs layout offsets its sticky header.
  changeLayout: boolean;
  className?: string;
  children: ReactNode;
};

const AnnouncementBar: React.FC<AnnouncementBarProps> = ({
  id,
  height,
  changeLayout,
  className,
  children,
}) => {
  const dismissedClass = `nd-banner-${id}`;
  const [open, setOpen] = useState(true);

  useEffect(() => {
    try {
      setOpen(localStorage.getItem(dismissedClass) !== "true");
    } catch {
      // Storage blocked: keep the bar open.
    }
  }, [dismissedClass]);

  if (!open) return null;

  const close = () => {
    setOpen(false);
    document.documentElement.classList.add(dismissedClass);
    try {
      localStorage.setItem(dismissedClass, "true");
    } catch {
      // Storage blocked: the bar stays closed until the next load.
    }
  };

  return (
    <div
      id={id}
      className={twMerge(
        "sticky top-0 z-40 flex flex-row items-center justify-center px-4 text-center text-sm font-medium bg-fd-secondary",
        className
      )}
      style={{ height }}
    >
      {changeLayout ? (
        <style>{`:root:not(.${dismissedClass}) { --fd-banner-height: ${height}; }`}</style>
      ) : null}
      <style>{`.${dismissedClass} #${id} { display: none; }`}</style>
      {children}
      <button
        type="button"
        aria-label="Close Banner"
        onClick={close}
        className={buttonVariants({
          color: "ghost",
          size: "icon-sm",
          className: "absolute end-2 top-1/2 -translate-y-1/2 text-fd-muted-foreground/50",
        })}
      >
        <X />
      </button>
    </div>
  );
};

export default AnnouncementBar;
