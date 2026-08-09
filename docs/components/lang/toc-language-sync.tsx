"use client";

import { useEffect } from "react";
import { useLanguage } from "@/components/lang/language-provider";

/**
 * Hides table-of-contents entries whose heading the current language does not
 * render.
 *
 * The TOC is derived from the MDX at build time, so it lists every heading on
 * the page — including ones inside `<Only>` or `<Switch>` that the reader's
 * language omits. Those would otherwise be links that scroll nowhere. Matching
 * on the anchor's presence in the DOM keeps this independent of which component
 * did the omitting.
 */
export const TocLanguageSync = () => {
  const { language } = useLanguage();

  useEffect(() => {
    // Runs after paint so conditional sections have already mounted/unmounted.
    const frame = requestAnimationFrame(() => {
      const links = document.querySelectorAll<HTMLAnchorElement>(
        '#nd-toc a[href^="#"], #nd-tocnav a[href^="#"]',
      );
      for (const link of links) {
        const id = decodeURIComponent(link.hash.slice(1));
        link.toggleAttribute("hidden", !id || !document.getElementById(id));
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [language]);

  return null;
};
