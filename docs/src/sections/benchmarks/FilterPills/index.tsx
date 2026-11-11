"use client";

import styles from "./FilterPills.module.scss";

export type PillOption<T extends string> = {
  id: T;
  label: string;
  count?: string | number;
};

type FilterPillsProps<T extends string> = {
  options: PillOption<T>[];
  value: T;
  onChange: (value: T) => void;
  label: string;
};

// A single-choice row of pill buttons.
function FilterPills<T extends string>({
  options,
  value,
  onChange,
  label,
}: FilterPillsProps<T>) {
  return (
    <div className={styles.pills} role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          role="radio"
          aria-checked={option.id === value}
          className={styles.pill}
          onClick={() => onChange(option.id)}
        >
          {option.label}
          {option.count !== undefined && (
            <span className={styles.count}>{option.count}</span>
          )}
        </button>
      ))}
    </div>
  );
}

export default FilterPills;
