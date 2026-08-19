"use client";

import type { ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";
import type { Language } from "@/lib/lang/languages";

/**
 * Content only one language gets, rendered for that language and no one else.
 * Every one-sided block is an Only, whether the gap is permanent or the other
 * SDK is still catching up. `<Switch>` is for content each language has its own
 * version of.
 */
export const Only = ({
  id,
  children,
}: {
  id: Language;
  children: ReactNode;
}) => {
  const { language } = useLanguage();
  return language === id ? <>{children}</> : null;
};
