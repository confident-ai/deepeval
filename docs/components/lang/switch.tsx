"use client";

import {
  Children,
  Fragment,
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

const isCase = (node: ReactNode): node is ReactElement<CaseProps> =>
  isValidElement(node) && isLanguage((node.props as CaseProps)?.id);

const isBlank = (node: ReactNode) =>
  typeof node === "string" && node.trim() === "";

/**
 * Renders the `<Case>` matching the reader's language, leaving everything else
 * in place.
 *
 * Children that are not a `<Case>` are shared — they render for every language,
 * in their original position. So the common part of a block is written once and
 * only the difference goes in a case: a `<Cards>` grid where a single card
 * differs, or a paragraph introducing two snippets.
 *
 * With no matching case — a feature one SDK doesn't cover yet — the first case
 * is shown under a notice naming what the reader is actually looking at, which
 * is the whole point of wrapping a lone block rather than leaving it bare. For
 * a gap that will never close, use `<Only>`, which renders nothing instead of
 * promising parity.
 *
 * Renders no wrapper element, so a `<Switch>` placed directly inside a `<Cards>`
 * grid or other layout container yields its cases as children of that container.
 */
export const Switch = ({ children }: { children: ReactNode }) => {
  const { language } = useLanguage();

  const items = Children.toArray(children).filter((node) => !isBlank(node));
  const cases = items.filter(isCase);
  if (cases.length === 0) return <>{items}</>;

  const unmatched = cases.every((entry) => entry.props.id !== language);
  const fallback = unmatched ? cases[0] : undefined;

  return (
    <>
      {items.map((item, index) => {
        if (!isCase(item)) return item;
        if (item.props.id === language) return item;
        if (item !== fallback) return null;
        return (
          <Fragment key={index}>
            <p className="text-sm text-fd-muted-foreground -mb-2">
              Shown in {LANGUAGES[item.props.id].label} — not yet available in
              the {LANGUAGES[language].label} SDK.
            </p>
            {item}
          </Fragment>
        );
      })}
    </>
  );
};
