"use client";

import type { ReactNode } from "react";
import styles from "./ViewToggle.module.scss";

export type ViewOption<T extends string> = {
  id: T;
  label: string;
  icon: ReactNode;
};

type ViewToggleProps<T extends string> = {
  options: ViewOption<T>[];
  value: T;
  onChange: (value: T) => void;
};

// Switches one section between two or more ways of showing the same data.
function ViewToggle<T extends string>({
  options,
  value,
  onChange,
}: ViewToggleProps<T>) {
  return (
    <div className={styles.toggle} role="group" aria-label="View">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          aria-pressed={option.id === value}
          onClick={() => onChange(option.id)}
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  );
}

export default ViewToggle;
