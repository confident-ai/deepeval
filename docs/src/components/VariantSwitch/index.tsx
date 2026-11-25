"use client";

import React, { useState } from "react";
import styles from "./VariantSwitch.module.scss";

interface VariantProps {
  value: string;
  children?: React.ReactNode;
}

/** One option of a VariantSwitch; only the selected one is shown. */
export const Variant: React.FC<VariantProps> = ({ children }) => <>{children}</>;

interface VariantSwitchProps {
  items: string[];
  children?: React.ReactNode;
}

/** A small segmented switch that swaps what's below it, such as a diagram's
 * stereo and mono versions, without the weight of a full tab group. */
const VariantSwitch: React.FC<VariantSwitchProps> = ({ items, children }) => {
  const [selected, setSelected] = useState(items[0]);
  const variants = React.Children.toArray(children).filter(
    (child): child is React.ReactElement<VariantProps> =>
      React.isValidElement(child) &&
      typeof (child.props as VariantProps).value === "string",
  );
  return (
    <div className={styles.switchBlock}>
      <div className={styles.switch} role="radiogroup">
        {items.map((item) => (
          <button
            key={item}
            type="button"
            role="radio"
            aria-checked={item === selected}
            className={item === selected ? styles.active : styles.option}
            onClick={() => setSelected(item)}
          >
            {item}
          </button>
        ))}
      </div>
      {variants.filter((variant) => variant.props.value === selected)}
    </div>
  );
};

export default VariantSwitch;
