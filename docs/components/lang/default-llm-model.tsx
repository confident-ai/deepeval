"use client";

import { useLanguage } from "@/components/lang/language-provider";
import { DEFAULT_LLM_MODEL } from "@/lib/defaults";

export const DefaultLLMModel = () => {
  const { language } = useLanguage();
  return <code>{DEFAULT_LLM_MODEL[language]}</code>;
};
