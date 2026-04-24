"use client";

import { useEffect } from "react";
import { Command } from "lucide-react";
import styles from "./Hotkey.module.scss";

export type HotkeyConfig = {
  key: string;
  action: () => void;
};

type HotkeyProps = {
  hotkey: HotkeyConfig;
  ariaLabel?: string;
};

const Hotkey: React.FC<HotkeyProps> = ({ hotkey, ariaLabel }) => {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT")
      ) {
        return;
      }

      if (event.repeat) return;
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() !== hotkey.key.toLowerCase()) return;

      event.preventDefault();
      hotkey.action();
    }

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [hotkey]);

  return (
    <kbd
      className={styles.root}
      aria-label={ariaLabel ?? `Command plus ${hotkey.key}`}
    >
      <span className={styles.icon} aria-hidden="true">
        <Command />
      </span>
      <span className={styles.key}>{hotkey.key}</span>
    </kbd>
  );
};


export default Hotkey;
