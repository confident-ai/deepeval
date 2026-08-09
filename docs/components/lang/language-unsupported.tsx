"use client";

import { useLanguage } from "@/components/lang/language-provider";
import { DinoGame } from "@/components/lang/dino-game";
import { LANGUAGES, type Language } from "@/lib/lang/languages";
import styles from "./language-unsupported.module.scss";

export const LanguageUnsupported = ({
  requested,
  supported,
}: {
  requested: Language;
  supported: Language[];
}) => {
  const { setLanguage } = useLanguage();
  const fallback = supported[0];

  return (
    // Hooks the global rule that hides the table of contents for this page.
    <div className={styles.wrapper} data-language-unsupported>
      <p className={styles.code}>501</p>
      <p className={styles.reason}>Not Implemented</p>
      <h1 className={styles.title}>
        Not available in {LANGUAGES[requested].label}
      </h1>
      <p className={styles.body}>
        This feature exists, but the {LANGUAGES[requested].label} SDK has not
        implemented it yet. The page is available in{" "}
        {supported.map((id) => LANGUAGES[id].label).join(" and ")}.
      </p>

      <button
        type="button"
        className={styles.action}
        onClick={() => setLanguage(fallback)}
      >
        Read it in {LANGUAGES[fallback].label}
      </button>

      <div className={styles.game}>
        <DinoGame />
      </div>
      <p className={styles.hint}>Or stay a while. Press space to jump.</p>
    </div>
  );
};
