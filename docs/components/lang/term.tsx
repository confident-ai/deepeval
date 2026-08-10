"use client";

import { useLanguage } from "@/components/lang/language-provider";
import { TERM_PROPS, type TermProps } from "@/lib/lang/term";

/**
 * An inline code span whose text differs between the SDKs:
 * `<Term py="actual_output" ts="actualOutput"/>`. Keep the name capitalized —
 * MDX compiles a lowercase tag in the markdown flow to a literal DOM element
 * instead of a component lookup, so every span would render empty.
 */
export const Term = (props: TermProps) => {
  const { language } = useLanguage();
  return <code>{props[TERM_PROPS[language]]}</code>;
};
