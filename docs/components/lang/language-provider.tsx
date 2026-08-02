"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";
import type { Language } from "@/lib/lang/terms";

const LanguageContext = createContext<{
  language: Language;
  preference: Language;
  setLanguage: (lang: Language) => void;
  pythonOnlyPage: boolean;
  setPythonOnlyPage: (value: boolean) => void;
}>({
  language: "python",
  preference: "python",
  setLanguage: () => {},
  pythonOnlyPage: false,
  setPythonOnlyPage: () => {},
});

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
  const [preference, setLanguage] = useState<Language>("python");
  const [pythonOnlyPage, setPythonOnlyPage] = useState(false);
  const language: Language = pythonOnlyPage ? "python" : preference;
  return (
    <LanguageContext.Provider
      value={{ language, preference, setLanguage, pythonOnlyPage, setPythonOnlyPage }}
    >
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);
