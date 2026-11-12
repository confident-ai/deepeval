"use client";

import * as Popover from "@radix-ui/react-popover";
import { Check, ChevronDown } from "lucide-react";
import styles from "./FilterMenu.module.scss";

export type FilterOption = { id: string; label: string; count: number };
export type FilterGroup = {
  id: string;
  label: string;
  options: FilterOption[];
};

type FilterMenuProps = {
  label: string;
  groups: FilterGroup[];
  selected: string[];
  onChange: (selected: string[]) => void;
};

// A dropdown of grouped checkboxes. Any number of options can be ticked.
const FilterMenu: React.FC<FilterMenuProps> = ({
  label,
  groups,
  selected,
  onChange,
}) => {
  const options = groups.flatMap((group) => group.options);
  const chosen = options.filter((option) => selected.includes(option.id));
  const toggle = (id: string) =>
    onChange(
      selected.includes(id)
        ? selected.filter((other) => other !== id)
        : [...selected, id],
    );

  return (
    <Popover.Root>
      <Popover.Trigger
        className={styles.trigger}
        data-active={chosen.length > 0}
      >
        <span className={styles.name}>{label}</span>
        <span className={styles.value}>
          {chosen.length === 0
            ? "Any"
            : chosen.length === 1
              ? chosen[0].label
              : `${chosen.length} selected`}
        </span>
        <ChevronDown className={styles.chevron} />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={styles.menu} align="start" sideOffset={6}>
          <div className={styles.scroll}>
            {groups.map((group) => (
              <div key={group.id} className={styles.group}>
                <div className={styles.heading}>{group.label}</div>
                {group.options.map((option) => {
                  const checked = selected.includes(option.id);
                  return (
                    <button
                      key={option.id}
                      type="button"
                      role="menuitemcheckbox"
                      aria-checked={checked}
                      className={styles.option}
                      onClick={() => toggle(option.id)}
                    >
                      <span className={styles.box}>{checked && <Check />}</span>
                      <span className={styles.label}>{option.label}</span>
                      <span className={styles.count}>{option.count}</span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          <div className={styles.foot}>
            <span>
              {chosen.length === 0
                ? "Nothing ticked"
                : `${chosen.length} ticked`}
            </span>
            <button
              type="button"
              onClick={() => onChange([])}
              disabled={chosen.length === 0}
            >
              Clear
            </button>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
};

export default FilterMenu;
