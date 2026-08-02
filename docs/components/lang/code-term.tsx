"use client";

import { useLanguage } from "@/components/lang/language-provider";
import { getTerm, type TermId } from "@/lib/lang/terms";

export const C = ({ id }: { id: TermId }) => {
  const { language } = useLanguage();
  return <code>{getTerm(id, language)}</code>;
};
