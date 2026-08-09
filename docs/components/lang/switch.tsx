"use client";

import {
  Children,
  isValidElement,
  type ReactElement,
  type ReactNode,
} from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LANGUAGE_IDS, type Language } from "@/lib/lang/languages";

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
 * A Switch is for content every language has its own version of, so every
 * language gets a case and `validate-terms` fails the build otherwise. A
 * one-sided block is an `<Only>`.
 *
 * Children that are not a `<Case>` are shared — they render for every language,
 * in their original position, so the common part of a block is written once.
 *
 * Renders no wrapper element, so a `<Switch>` placed directly inside a `<Cards>`
 * grid or other layout container yields its cases as children of that container.
 */
export const Switch = ({ children }: { children: ReactNode }) => {
  const { language } = useLanguage();

  const items = Children.toArray(children).filter((node) => !isBlank(node));

  return (
    <>
      {items.map((item) => {
        if (!isCase(item)) return item;
        return item.props.id === language ? item : null;
      })}
    </>
  );
};
