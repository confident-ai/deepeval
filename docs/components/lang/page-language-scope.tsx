"use client";

import type { ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LanguageUnsupported } from "@/components/lang/language-unsupported";
import type { Language } from "@/lib/lang/languages";

/**
 * Gates a page on the reader's language.
 *
 * Content only ever renders in the language the reader actually picked, so a
 * page that cannot serve it shows nothing rather than quietly falling back to
 * the other language. The preference is left untouched so it still applies on
 * the next page.
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
