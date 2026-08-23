"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { Check, ChevronsUpDown } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "fumadocs-ui/components/ui/popover";
import { useLanguage } from "@/components/lang/language-provider";
import {
  LANGUAGES,
  LANGUAGE_IDS,
  type Language,
  type LanguageMeta,
} from "@/lib/lang/languages";
import { SDK_VERSIONS } from "@/lib/lang/versions";
import { StatusTag } from "@/components/status-tag";
import {
  LANGUAGE_SELECTOR_HINT_DISMISSED_EVENT,
  LANGUAGE_SELECTOR_HINT_STORAGE_KEY,
} from "./language-selector-hint";
import styles from "./language-selector.module.scss";

const badge = (src: string, label: string) => (
  <img
    className={styles.icon}
    src={src}
    alt={`${label} logo`}
    width={20}
    height={20}
  />
);

const tag = (text: string | undefined) =>
  text ? <StatusTag>{text}</StatusTag> : null;

type LanguageOption = { id: Language } & LanguageMeta;

const OPTIONS: LanguageOption[] = LANGUAGE_IDS.map((id) => ({
  id,
  ...LANGUAGES[id],
}));

// The reference surfaces. Guides and tutorials still honour the selection made
// here, they just don't offer their own control.
const SECTIONS = ["/docs", "/integrations"];

const LanguageSelector = () => {
  const { language, setLanguage } = useLanguage();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const showSelector = SECTIONS.some(
    (section) => pathname === section || pathname.startsWith(`${section}/`),
  );

  useEffect(() => {
    if (!showSelector || !window.matchMedia("(min-width: 768px)").matches) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      if (!triggerRef.current?.closest("#nd-sidebar")) return;

      try {
        if (window.localStorage.getItem(LANGUAGE_SELECTOR_HINT_STORAGE_KEY)) {
          return;
        }
      } catch {
        // Keep the first-visit experience available when storage is blocked.
      }

      setOpen(true);
    });

    return () => window.cancelAnimationFrame(frame);
  }, [showSelector]);

  if (!showSelector) {
    return null;
  }

  const active = OPTIONS.find((o) => o.id === language) ?? OPTIONS[0];

  const select = (option: LanguageOption) => {
    setLanguage(option.id);
    setOpen(false);
    try {
      window.localStorage.setItem(
        LANGUAGE_SELECTOR_HINT_STORAGE_KEY,
        "dismissed",
      );
    } catch {
      // Selection still dismisses the hint for the current page.
    }
    window.dispatchEvent(new Event(LANGUAGE_SELECTOR_HINT_DISMISSED_EVENT));
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        ref={triggerRef}
        aria-label="Select language"
        className={styles.trigger}
        data-language-selector=""
      >
        {badge(active.icon, active.label)}
        <span className={styles.label}>{active.label}</span>
        {tag(active.tag)}
        <span className={styles.version}>{`v${SDK_VERSIONS[active.id]}`}</span>
        <ChevronsUpDown className={styles.chevron} />
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className={`${styles.content} z-[110]`}
        data-language-selector-dropdown=""
      >
        {OPTIONS.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => select(option)}
            className={styles.item}
          >
            <div className={styles.itemContent}>
              {badge(option.icon, option.label)}
              <span className={styles.text}>
                <span className={styles.labelRow}>
                  <span className={styles.label}>{option.label}</span>
                  {tag(option.tag)}
                </span>
                <span className={styles.description}>{option.description}</span>
              </span>
            </div>
            <span className={styles.version}>{`v${SDK_VERSIONS[option.id]}`}</span>
            <Check
              className={`${styles.check} ${
                option.id === active.id ? "" : styles.hidden
              }`}
            />
          </button>
        ))}
      </PopoverContent>
    </Popover>
  );
};

export default LanguageSelector;
