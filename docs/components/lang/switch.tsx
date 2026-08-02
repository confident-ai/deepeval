"use client";

import {
  Children,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LANGUAGES, LANGUAGE_IDS, type Language } from "@/lib/lang/languages";

type CaseProps = { id: Language; children: ReactNode };

/** One language's version of a block. Only meaningful inside Switch. */
export const Case = ({ children }: CaseProps) => <>{children}</>;

const isLanguage = (value: unknown): value is Language =>
  LANGUAGE_IDS.includes(value as Language);

/**
 * Renders the `<Case>` matching the reader's language.
 *
 * With no match — a feature one SDK doesn't cover yet — the first case is shown
 * under a notice naming what the reader is actually looking at, which is the
 * whole point of wrapping a lone block rather than leaving it bare.
 */
export const Switch = ({ children }: { children: ReactNode }) => {
  const { language } = useLanguage();

  const cases = Children.toArray(children)
    .filter(isValidElement)
    .map((child) => child as ReactElement<CaseProps>)
    .filter((child) => isLanguage(child.props?.id));

  const match = cases.find((entry) => entry.props.id === language);
  if (match) return match;

  const [fallback] = cases;
  if (!fallback) return null;

  return (
    <>
      <p className="text-sm text-fd-muted-foreground -mb-2">
        Shown in {LANGUAGES[fallback.props.id].label} — not yet available in the{" "}
        {LANGUAGES[language].label} SDK.
      </p>
      {fallback}
    </>
  );
};
