"use client";

import { useEffect } from "react";

// fumadocs only adds `nd-banner-<id>` to <html> on page load, not when the
// close button is clicked. The meetup CSS keys the docs offset
// (--fd-banner-height) on that class, so add it on click too; otherwise a
// closed bar would leave a 30px gap until the next full load.
const BannerDismissSync: React.FC = () => {
  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      const closeButton = (event.target as Element | null)?.closest?.(
        'button[aria-label="Close Banner"]'
      );
      const bannerId = closeButton?.parentElement?.id;
      if (bannerId) document.documentElement.classList.add(`nd-banner-${bannerId}`);
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, []);

  return null;
};

export default BannerDismissSync;
