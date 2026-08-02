"use client";

import { Children, isValidElement, type ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";

/**
 * Selection is positional and unvalidated: child 0 must be Python, child 1
 * TypeScript. Reversing them fails silently by serving the wrong language.
 *
 * A single child is deliberate, not an unfinished block — it marks a feature
 * the TypeScript SDK does not cover yet on an otherwise bilingual page, and
 * is the correct alternative to leaving a bare ```python fence outside a
 * LangSwitch, which gives the reader no signal at all.
 */
export const LangSwitch = ({ children }: { children: ReactNode }) => {
  const { language } = useLanguage();
  const blocks = Children.toArray(children).filter(isValidElement);

  if (blocks.length === 0) return null;

  const wantsTypescript = language === "typescript";
  if (wantsTypescript && blocks.length === 1) {
    return (
      <>
        <p className="text-sm text-fd-muted-foreground -mb-2">
          Shown in Python — not yet available in the TypeScript SDK.
        </p>
        {blocks[0]}
      </>
    );
  }

  return <>{blocks[wantsTypescript ? 1 : 0]}</>;
};
