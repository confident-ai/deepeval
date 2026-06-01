"use client";

import {
  useEffect,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";

/* --------------------------------------------------------------------
 * PauseOffscreen
 *
 * Wraps a chunk of UI in a div whose `data-paused` attribute toggles
 * based on whether the element is in the viewport. Combined with the
 * global rule in `docs/app/global.css`:
 *
 *   [data-paused="true"], [data-paused="true"] * {
 *     animation-play-state: paused !important;
 *   }
 *
 * …all CSS animations on the wrapped subtree freeze when scrolled out
 * of view, dropping the GPU/compositor cost to ~0 for offscreen
 * sections. This is the cheap fix for the home-page scroll lag on
 * iPad / lower-spec laptops, where 30+ infinite SVG animations were
 * always running regardless of viewport visibility.
 *
 * SSR-safe (does nothing until mounted) and degrades cleanly on
 * browsers without IntersectionObserver (animations just keep running
 * the way they used to).
 *
 * Extra HTMLAttributes are spread onto the wrapper div so the
 * component can drop into existing layouts as the host element
 * (carrying className + aria-label etc.) instead of nesting an
 * extra div between layout-significant parents and children.
 * ------------------------------------------------------------------ */

type Props = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
  /**
   * IntersectionObserver `rootMargin`. Default `200px` starts
   * resuming animations a touch before they enter the viewport so
   * the user never sees a frozen frame as they scroll in.
   */
  rootMargin?: string;
};

export const PauseOffscreen: React.FC<Props> = ({
  children,
  rootMargin = "200px",
  ...rest
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;

    const io = new IntersectionObserver(
      ([entry]) => setPaused(!entry.isIntersecting),
      { rootMargin }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [rootMargin]);

  return (
    <div ref={ref} data-paused={paused ? "true" : undefined} {...rest}>
      {children}
    </div>
  );
};

export default PauseOffscreen;
