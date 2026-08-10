"use client";

import type { ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LanguageUnsupported } from "@/components/lang/language-unsupported";
import type { Language } from "@/lib/lang/languages";

/**
 * Gates a page on the reader's language.
 *
 * Content only ever renders in a language the page declares, so a mismatch
 * shows a 501 rather than quietly falling back. In practice
 * `LanguageProvider` already opens mono-language pages in their only
 * language (SSR and soft-nav), so this is a safety net for odd edge cases.
 * Bilingual pages keep the reader's preference across navigations.
 */
export const PageLanguageScope = ({
  languages,
  children,
}: {
  languages?: Language[];
  children: ReactNode;
}) => {
  const { language } = useLanguage();

  if (languages?.length && !languages.includes(language)) {
    return <LanguageUnsupported requested={language} supported={languages} />;
  }

  return <>{children}</>;
};
