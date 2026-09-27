"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import katex from "katex";
import styles from "./Equation.module.scss";

interface EquationProps {
  formula: string;
}

/**
 * A formula is one unbreakable line, so a long one scrolls horizontally. The
 * edge fades exist to advertise that: they appear only on the side that has
 * more formula hidden behind it, so a formula that fits shows none at all.
 */
const Equation: React.FC<EquationProps> = (props) => {
  const html = katex.renderToString(props.formula, {
    throwOnError: false,
    displayMode: true,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const [overflow, setOverflow] = useState({ left: false, right: false });

  const syncOverflow = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    // A pixel of slack: fractional scroll offsets would otherwise leave the
    // fade on at either end of the range.
    const scrollableWidth = container.scrollWidth - container.clientWidth;
    setOverflow({
      left: container.scrollLeft > 1,
      right: container.scrollLeft < scrollableWidth - 1,
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    syncOverflow();
    // Whether a formula overflows depends on the column width, so this has to
    // be re-checked on resize rather than measured once on mount.
    const observer = new ResizeObserver(syncOverflow);
    observer.observe(container);
    return () => observer.disconnect();
  }, [syncOverflow, html]);

  const className = [
    styles.equationContainer,
    overflow.left ? styles.fadeLeft : "",
    overflow.right ? styles.fadeRight : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div ref={containerRef} className={className} onScroll={syncOverflow}>
      <span dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
};

export default Equation;
