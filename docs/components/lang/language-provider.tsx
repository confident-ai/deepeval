"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { Language } from "@/lib/lang/terms";

/**
 * Single shared source of truth for the active code language.
 *
 * Deliberately does NOT read the URL, pathname, or any route param —
 * routing is a separate decision for later. For now it just holds state
 * defaulting to Python; the `setLanguage` setter exists so a future
 * language dropdown can flip it, but nothing toggles it yet.
 */
const LanguageContext = createContext<{
  language: Language;
  setLanguage: (lang: Language) => void;
}>({
  language: "python",
  setLanguage: () => {},
});

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>("python");
  return (
    <LanguageContext.Provider value={{ language, setLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
