"use client";

import { useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { Check, ChevronsUpDown } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "fumadocs-ui/components/ui/popover";
import { useLanguage } from "@/components/lang/language-provider";
import type { Language } from "@/lib/lang/terms";
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

interface LanguageOption {
  id: Language;
  label: string;
  icon: ReactNode;
  description?: string;
  disabled?: boolean;
}

const OPTIONS: LanguageOption[] = [
  {
    id: "python",
    label: "Python",
    icon: badge("/icons/python.svg", "Python"),
    description: "First class support",
  },
  {
    id: "typescript",
    label: "TypeScript",
    icon: badge("/icons/typescript.svg", "TypeScript"),
    description: "Beta release",
  },
];

const LanguageSelector = () => {
  const { language, setLanguage, pythonOnlyPage } = useLanguage();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const showSelector =
    pathname === "/docs" ||
    pathname.startsWith("/docs/") ||
    pathname === "/integrations" ||
    pathname.startsWith("/integrations/");

  if (!showSelector) {
    return null;
  }

  const active = OPTIONS.find((o) => o.id === language) ?? OPTIONS[0];

  const isDisabled = (option: LanguageOption) =>
    option.disabled || (option.id === "typescript" && pythonOnlyPage);

  const description = (option: LanguageOption) =>
    option.id === "typescript" && pythonOnlyPage
      ? "Not available for this page"
      : option.description;

  const select = (option: LanguageOption) => {
    if (isDisabled(option)) return;
    setLanguage(option.id);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger aria-label="Select language" className={styles.trigger}>
        {active.icon}
        <span className={styles.label}>{active.label}</span>
        <ChevronsUpDown className={styles.chevron} />
      </PopoverTrigger>
      <PopoverContent align="start" className={styles.content}>
        {OPTIONS.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => select(option)}
            aria-disabled={isDisabled(option)}
            className={`${styles.item} ${
              isDisabled(option) ? styles.disabled : ""
            }`}
          >
            <div className={styles.itemContent}>
              {option.icon}
              <span className={styles.text}>
                <span className={styles.label}>{option.label}</span>
                {description(option) ? (
                  <span className={styles.description}>
                    {description(option)}
                  </span>
                ) : null}
              </span>
            </div>
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
