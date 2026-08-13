"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { usePathname } from "next/navigation";
import styles from "./language-selector-hint.module.scss";

export const LANGUAGE_SELECTOR_HINT_STORAGE_KEY =
  "deepeval:language-selector-hint:v1";
export const LANGUAGE_SELECTOR_HINT_DISMISSED_EVENT =
  "deepeval:language-selector-hint-dismissed";
const DESKTOP_QUERY = "(min-width: 768px)";
const ELIGIBLE_SECTIONS = ["/docs", "/integrations"];
const MASK_ID = "language-selector-hint-mask";
const ARROW_MARKER_ID = "language-selector-hint-arrow";

type HintLayout = {
  viewportWidth: number;
  viewportHeight: number;
  spotlightX: number;
  spotlightY: number;
  spotlightWidth: number;
  spotlightHeight: number;
  calloutLeft: number;
  calloutTop: number;
  arrowPath: string;
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

// Viewfinder-style brackets: only the four corners are drawn, so the
// highlight reads as "focused on this" rather than as a bordered box.
const cornerBrackets = (
  x: number,
  y: number,
  width: number,
  height: number
) => {
  const arm = Math.min(12, width / 4, height / 4);
  const right = x + width;
  const bottom = y + height;

  return [
    `M ${x} ${y + arm} L ${x} ${y} L ${x + arm} ${y}`,
    `M ${right - arm} ${y} L ${right} ${y} L ${right} ${y + arm}`,
    `M ${right} ${bottom - arm} L ${right} ${bottom} L ${
      right - arm
    } ${bottom}`,
    `M ${x + arm} ${bottom} L ${x} ${bottom} L ${x} ${bottom - arm}`,
  ].join(" ");
};

const LanguageSelectorHint = () => {
  const pathname = usePathname();
  const overlayRef = useRef<HTMLButtonElement>(null);
  // The resize/scroll listeners outlive a dismissal, so they need their own
  // record of it — re-reading storage on every frame would be wasteful, and
  // storage may be blocked entirely.
  const dismissedRef = useRef(false);
  const [layout, setLayout] = useState<HintLayout | null>(null);
  const isVisible = layout !== null;

  const eligible = ELIGIBLE_SECTIONS.some(
    (section) => pathname === section || pathname.startsWith(`${section}/`)
  );

  const dismiss = useCallback(() => {
    try {
      window.localStorage.setItem(
        LANGUAGE_SELECTOR_HINT_STORAGE_KEY,
        "dismissed"
      );
    } catch {
      // Storage can be unavailable in privacy modes; dismissal still works
      // for the current page.
    }
    dismissedRef.current = true;
    setLayout(null);
  }, []);

  useEffect(() => {
    if (!eligible) {
      setLayout(null);
      return;
    }

    if (dismissedRef.current) {
      setLayout(null);
      return;
    }

    try {
      if (window.localStorage.getItem(LANGUAGE_SELECTOR_HINT_STORAGE_KEY)) {
        dismissedRef.current = true;
        setLayout(null);
        return;
      }
    } catch {
      // A blocked localStorage should not prevent the hint from rendering.
    }

    const mediaQuery = window.matchMedia(DESKTOP_QUERY);

    const updateLayout = () => {
      if (dismissedRef.current) {
        setLayout(null);
        return;
      }

      if (!mediaQuery.matches) {
        setLayout(null);
        return;
      }

      const target = document.querySelector<HTMLElement>(
        "#nd-sidebar [data-language-selector]"
      );
      if (!target) {
        setLayout(null);
        return;
      }

      const rect = target.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        setLayout(null);
        return;
      }

      const dropdown = Array.from(
        document.querySelectorAll<HTMLElement>(
          "[data-language-selector-dropdown]"
        )
      ).find((element) => {
        const dropdownRect = element.getBoundingClientRect();
        return dropdownRect.width > 0 && dropdownRect.height > 0;
      });
      const dropdownRect = dropdown?.getBoundingClientRect();
      const spotlightLeft = Math.min(
        rect.left,
        dropdownRect?.left ?? rect.left
      );
      const spotlightTop = Math.min(rect.top, dropdownRect?.top ?? rect.top);
      const spotlightRight = Math.max(
        rect.right,
        dropdownRect?.right ?? rect.right
      );
      const spotlightBottom = Math.max(
        rect.bottom,
        dropdownRect?.bottom ?? rect.bottom
      );

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const spotlightPadding = 8;
      const calloutWidth = 300;
      const calloutLeft = clamp(
        spotlightRight + 132,
        24,
        Math.max(24, viewportWidth - calloutWidth - 24)
      );
      const calloutTop = clamp(
        spotlightBottom + 64,
        24,
        Math.max(24, viewportHeight - 104)
      );

      const arrowStartX = calloutLeft + 24;
      const arrowStartY = calloutTop - 20;
      // Keep the arrowhead just outside the trigger's right edge. Pointing
      // underneath the open popover hid the arrowhead and made the visible
      // curve look like it was aimed at the callout instead.
      const arrowEndX = rect.right + spotlightPadding + 10;
      const arrowEndY = rect.top + rect.height / 2;
      // Bows out to the right before hooking back into the trigger, which
      // reads as a hand-drawn curve instead of a straight connector.
      const arrowPath = [
        `M ${arrowStartX} ${arrowStartY}`,
        `C ${arrowStartX + 52} ${arrowStartY - 58},`,
        `${arrowEndX + 128} ${arrowEndY + 46},`,
        `${arrowEndX} ${arrowEndY}`,
      ].join(" ");

      setLayout({
        viewportWidth,
        viewportHeight,
        spotlightX: spotlightLeft - spotlightPadding,
        spotlightY: spotlightTop - spotlightPadding,
        spotlightWidth: spotlightRight - spotlightLeft + spotlightPadding * 2,
        spotlightHeight: spotlightBottom - spotlightTop + spotlightPadding * 2,
        calloutLeft,
        calloutTop,
        arrowPath,
      });
    };

    const frame = window.requestAnimationFrame(updateLayout);
    const retry = window.setTimeout(updateLayout, 200);
    window.addEventListener("resize", updateLayout);
    window.addEventListener("scroll", updateLayout, true);
    mediaQuery.addEventListener("change", updateLayout);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(retry);
      window.removeEventListener("resize", updateLayout);
      window.removeEventListener("scroll", updateLayout, true);
      mediaQuery.removeEventListener("change", updateLayout);
    };
  }, [eligible, pathname]);

  useEffect(() => {
    const dismissFromSelector = () => {
      dismissedRef.current = true;
      setLayout(null);
    };
    window.addEventListener(
      LANGUAGE_SELECTOR_HINT_DISMISSED_EVENT,
      dismissFromSelector
    );
    return () =>
      window.removeEventListener(
        LANGUAGE_SELECTOR_HINT_DISMISSED_EVENT,
        dismissFromSelector
      );
  }, []);

  useEffect(() => {
    if (!isVisible) return;

    overlayRef.current?.focus({ preventScroll: true });

    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismiss();
    };
    document.addEventListener("keydown", dismissOnEscape);
    return () => document.removeEventListener("keydown", dismissOnEscape);
  }, [dismiss, isVisible]);

  if (!layout) return null;

  return createPortal(
    <button
      ref={overlayRef}
      type="button"
      className={styles.overlay}
      aria-label="Dismiss language selector tip"
      onClick={dismiss}
    >
      <svg
        className={styles.canvas}
        viewBox={`0 0 ${layout.viewportWidth} ${layout.viewportHeight}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          <mask id={MASK_ID}>
            <rect
              width={layout.viewportWidth}
              height={layout.viewportHeight}
              fill="white"
            />
            <rect
              x={layout.spotlightX}
              y={layout.spotlightY}
              width={layout.spotlightWidth}
              height={layout.spotlightHeight}
              rx="6"
              fill="black"
            />
          </mask>
          <marker
            id={ARROW_MARKER_ID}
            markerWidth="14"
            markerHeight="14"
            refX="11"
            refY="7"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M 2 2 L 12 7 L 2 12" className={styles.arrowHead} />
          </marker>
        </defs>
        <rect
          width={layout.viewportWidth}
          height={layout.viewportHeight}
          className={styles.backdrop}
          mask={`url(#${MASK_ID})`}
        />
        <path
          d={cornerBrackets(
            layout.spotlightX,
            layout.spotlightY,
            layout.spotlightWidth,
            layout.spotlightHeight
          )}
          className={styles.spotlight}
        />
        <path
          d={layout.arrowPath}
          className={styles.arrow}
          markerEnd={`url(#${ARROW_MARKER_ID})`}
        />
      </svg>
      <span
        className={styles.callout}
        style={{ left: layout.calloutLeft, top: layout.calloutTop }}
      >
        DeepEval is now available in both Python and TypeScript :)
      </span>
      <span className={styles.dismissal}>Click anywhere to continue</span>
    </button>,
    document.body
  );
};

export default LanguageSelectorHint;
