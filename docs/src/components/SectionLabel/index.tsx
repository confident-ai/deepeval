import type { ReactNode } from "react";
import styles from "./SectionLabel.module.scss";

type SectionLabelProps = {
  children: ReactNode;
};

const SectionLabel: React.FC<SectionLabelProps> = ({ children }) => {
  return <p className={styles.label}>{children}</p>;
};

export default SectionLabel;
