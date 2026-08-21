"use client";

import type { ReactNode } from "react";
import { useLanguage } from "@/components/lang/language-provider";
import { LANGUAGES, LANGUAGE_IDS, type Language } from "@/lib/lang/languages";
import styles from "./not-implemented.module.scss";

/**
 * A section the named language's SDK has not implemented yet. The section-level
 * counterpart to the 501 page a whole `[python]` page shows: the heading stays,
 * so the reader learns the feature exists and is simply not theirs yet, rather
 * than never seeing it at all.
 *
 * It goes inside the `<Case>` for the language that lacks the feature, standing
 * in for the content that case will eventually hold. Closing the gap is then
 * filling that case in — delete the notice, write the snippet — rather than
 * restructuring an `<Only>` into a `<Switch>` first.
 *
 * Reserved for gaps that are expected to close. Something with no possible
 * counterpart — a Python-only framework, notebooks — is a plain `<Only>`, since
 * promising it is coming would be a lie. That distinction is what makes counting
 * these tags a parity backlog:
 *
 *     rg -c '<NotImplemented' content
 *
 * The `id` is still required even though the enclosing case has fixed the
 * language: it keeps the count filterable per SDK, and makes the notice render
 * for that language only if it is ever written outside a case.
 */
export const NotImplemented = ({
  id,
  feature,
  children,
}: {
  /** The language that lacks the feature. */
  id: Language;
  /**
   * The missing API, spelled as the SDK that has it spells it. Rendered as a
   * code span, so pass the bare identifier — markdown in a string prop would
   * render its own backticks.
   */
  feature: string;
  /** What to do in the meantime. */
  children?: ReactNode;
}) => {
  const { language, setLanguage } = useLanguage();
  if (language !== id) return null;

  const elsewhere = LANGUAGE_IDS.filter((other) => other !== id);

  return (
    <aside className={styles.notice}>
      <div className={styles.header}>
        <span className={styles.code}>501</span>
        <span className={styles.title}>
          Not implemented in {LANGUAGES[id].label}
        </span>
      </div>

      <div className={styles.body}>
        <p className={styles.lead}>
          <code>{feature}</code> has not landed in the {LANGUAGES[id].label} SDK
          yet.
          {elsewhere.length === 1 ? (
            <>
              {" "}
              It is available in {LANGUAGES[elsewhere[0]].label} today, and this
              section is written for it.
            </>
          ) : null}
        </p>
        {children}
      </div>

      {elsewhere.length === 1 ? (
        <button
          type="button"
          className={styles.action}
          onClick={() => setLanguage(elsewhere[0])}
        >
          Read it in {LANGUAGES[elsewhere[0]].label}
        </button>
      ) : null}
    </aside>
  );
};
