import React from "react";
import {
  Info,
  Lightbulb,
  StickyNote,
  TriangleAlert,
  CircleAlert,
  CircleCheck,
  Bookmark,
} from "lucide-react";
import styles from "./Callout.module.scss";

export type CalloutType =
  | "note"
  | "info"
  | "tip"
  | "success"
  | "important"
  | "warning"
  | "caution"
  | "danger"
  | "error"
  | "secondary";

interface CalloutProps {
  type?: CalloutType;
  title?: React.ReactNode;
  children?: React.ReactNode;
}

const ICONS: Record<CalloutType, React.ComponentType<{ className?: string }>> = {
  note: StickyNote,
  info: Info,
  tip: Lightbulb,
  success: CircleCheck,
  important: Bookmark,
  warning: TriangleAlert,
  caution: TriangleAlert,
  danger: CircleAlert,
  error: CircleAlert,
  secondary: StickyNote,
};

const DEFAULT_TITLES: Partial<Record<CalloutType, string>> = {
  note: "Note",
  info: "Info",
  tip: "Tip",
  success: "Success",
  important: "Important",
  warning: "Warning",
  caution: "Caution",
  danger: "Danger",
  error: "Error",
};

const Callout: React.FC<CalloutProps> = ({ type = "note", title, children }) => {
  const Icon = ICONS[type] ?? StickyNote;
  const displayTitle = title ?? DEFAULT_TITLES[type];

  return (
    <aside className={styles.callout} data-type={type}>
      <div className={styles.header}>
        <Icon className={styles.icon} />
        {displayTitle ? <span className={styles.title}>{displayTitle}</span> : null}
      </div>
      <div className={styles.body}>{children}</div>
    </aside>
  );
};

export default Callout;
