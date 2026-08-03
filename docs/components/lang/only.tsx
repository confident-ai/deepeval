"use client";

import type { ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";
import type { Language } from "@/lib/lang/languages";

/**
 * Content for one language that the others are never getting.
 *
 * A one-case `<Switch>` says "not yet" and shows the block under a notice,
 * which is right while an SDK is catching up. This is for the other kind of
 * gap — a feature with no counterpart to wait for — where the honest render is
 * nothing at all.
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
