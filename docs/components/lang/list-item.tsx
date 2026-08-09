"use client";

import {
  Children,
  isValidElement,
  type ComponentProps,
  type ReactNode,
} from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LANGUAGES, type Language } from "@/lib/lang/languages";

/**
 * The language a bullet belongs to, when its whole content is one-sided.
 *
 * Markdown closes the `<ul>` at a block-level JSX tag, so wrapping the item's
 * content inline is the only way to vary one bullet without splitting the list
 * in two. That leaves an empty `<li>` — a bare bullet marker — for every other
 * language, and CSS cannot hide it: MDX pads a list item's JSX child with
 * newline text nodes, so `li:empty` never matches.
 *
 * The bullet is matched on its sole child carrying a `Language`-valued `id`
 * rather than on the `Only` component itself, because a server-rendered page
 * hands `li` a client reference whose identity need not be the `Only` this
 * module imports. Descending through sole wrappers covers a loose list (remark
 * wraps every item's content in a `p`) and inline emphasis around the tag.
 */
function oneSidedLanguage(children: ReactNode): Language | undefined {
  const meaningful = Children.toArray(children).filter(
    (child) => typeof child !== "string" || child.trim() !== "",
  );
  if (meaningful.length !== 1) return undefined;

  const [child] = meaningful;
  if (!isValidElement(child)) return undefined;

  const { id, children: nested } = child.props as {
    id?: unknown;
    children?: ReactNode;
  };
  if (typeof id === "string" && id in LANGUAGES) return id as Language;
  return oneSidedLanguage(nested);
}

/**
 * A markdown list item, dropped entirely when it belongs to another language.
 * Registered as the `li` element so authors can write ordinary bullets:
 *
 * ```mdx
 * - Shared by both.
 * - <Only id="python">Python has no counterpart elsewhere.</Only>
 * ```
 */
export const ListItem = (props: ComponentProps<"li">) => {
  const { language } = useLanguage();
  const owner = oneSidedLanguage(props.children);
  return owner && owner !== language ? null : <li {...props} />;
};
