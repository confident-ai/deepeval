import type { ReactNode } from "react";
import ProviderMark from "../ProviderMark";
import styles from "./ModelName.module.scss";

type ModelNameProps = {
  model: string;
  provider: string;
  size?: "sm" | "md";
  // Rendered after the name, e.g. a trophy or a tag.
  children?: ReactNode;
};

// A model's name with its provider's logo. Use this wherever a model is
// named so the logo is always present and always the same.
const ModelName: React.FC<ModelNameProps> = ({
  model,
  provider,
  size = "sm",
  children,
}) => {
  return (
    <span className={styles.modelName}>
      <ProviderMark provider={provider} size={size} />
      <span className={styles.text}>{model}</span>
      {children}
    </span>
  );
};

export default ModelName;
