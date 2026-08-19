"use client";

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { usePathname } from "next/navigation";
import {
  DEFAULT_LANGUAGE,
  resolveInitialLanguage,
  type Language,
} from "@/lib/lang/languages";
import { getPageLanguages } from "@/lib/lang/page-languages";

const LanguageContext = createContext<{
  language: Language;
  setLanguage: (lang: Language) => void;
}>({
  language: DEFAULT_LANGUAGE,
  setLanguage: () => {},
});

/**
 * Holds the reader's language selection for the session.
 *
 * Initial value (and SSR HTML) comes from the current page's `languages`
 * frontmatter: mono-language pages open in their only language so a TS-only
 * URL never paints a Python 501 for crawlers. Bilingual / undeclared pages
 * still default to Python. Soft-navigating onto a mono-language page adopts
 * that language; bilingual pages keep the current preference.
 */
export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  const pathname = usePathname();
  const pageLangs = getPageLanguages(pathname);
  const pageDefault = resolveInitialLanguage(pageLangs);

  const [language, setLanguage] = useState<Language>(pageDefault);
  const [prevPathname, setPrevPathname] = useState(pathname);

  if (pathname !== prevPathname) {
    setPrevPathname(pathname);
    const nextLangs = getPageLanguages(pathname);
    if (nextLangs?.length === 1) {
      setLanguage(nextLangs[0]);
    }
  }

  const value = useMemo(() => ({ language, setLanguage }), [language]);
  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);
