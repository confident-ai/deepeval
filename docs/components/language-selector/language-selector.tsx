"use client";

import { useState, type ReactNode } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "fumadocs-ui/components/ui/popover";
import { useLanguage } from "@/components/lang/language-provider";
import type { Language } from "@/lib/lang/terms";
import styles from "./language-selector.module.scss";

function badge(id: Language, src: string, label: string) {
  return (
    <img
      className={styles.icon}
      src={src}
      alt={`${label} logo`}
      width={20}
      height={20}
    />
  );
}

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
    icon: badge("python", "/icons/python.svg", "Python"),
    description: "First class support",
  },
  {
    id: "typescript",
    label: "TypeScript",
    icon: badge("typescript", "/icons/typescript.svg", "TypeScript"),
    description: "Coming soon on July 1st",
    disabled: true,
  },
];

export default function LanguageSelector() {
  const { language, setLanguage } = useLanguage();
  const [open, setOpen] = useState(false);

  const active = OPTIONS.find((o) => o.id === language) ?? OPTIONS[0];

  // TypeScript (and any disabled option) is a no-op for now.
  const select = (option: LanguageOption) => {
    if (option.disabled) return;
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
            aria-disabled={option.disabled}
            className={`${styles.item} ${
              option.disabled ? styles.disabled : ""
            }`}
          >
            <div className={styles.itemContent}>
              {option.icon}
              <span className={styles.text}>
                <span className={styles.label}>{option.label}</span>
                {option.description ? (
                  <span className={styles.description}>
                    {option.description}
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
}
