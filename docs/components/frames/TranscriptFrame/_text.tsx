import type * as React from "react";

/**
 * Turn text with ==highlighted== spans (markdown's highlight syntax, so
 * asterisks in quoted prompts such as "*****" stay literal).
 */
export const highlightParts = (text: string): { text: string; highlight: boolean }[] =>
  text
    .split(/(==[^=]+==)/)
    .filter(Boolean)
    .map(part =>
      part.startsWith("==") && part.endsWith("==") && part.length > 4
        ? { text: part.slice(2, -2), highlight: true }
        : { text: part, highlight: false }
    );

export const plainText = (text: string) => highlightParts(text).map(p => p.text).join("");

export const renderHighlights = (text: string, className: string): React.ReactNode =>
  highlightParts(text).map((part, i) =>
    part.highlight ? (
      <mark key={i} className={className}>
        {part.text}
      </mark>
    ) : (
      part.text
    )
  );
