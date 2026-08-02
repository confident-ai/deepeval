"use client";

import { Children, isValidElement, type ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";

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
