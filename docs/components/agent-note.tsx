import type { ReactNode } from "react";
import { Bot } from "lucide-react";
import styles from "@site/src/components/Callout/Callout.module.scss";

/**
 * Setup note for AI coding agents, same for every visitor.
 *
 * Usage in MDX: `<AgentNote tag="de-intro" />`. The text and the copyable
 * prompt are filled in at build time by `lib/remark-agent-note.ts`, so they
 * also appear in the page's llms markdown output.
 */
export function AgentNote({ tag, children }: { tag: string; children?: ReactNode }) {
  return (
    <aside className={styles.callout} data-type="secondary" data-agent-note={tag}>
      <div className={styles.header}>
        <Bot className={styles.icon} />
        <span className={styles.title}>For AI agents</span>
      </div>
      <div className={styles.body}>{children}</div>
    </aside>
  );
}

export default AgentNote;
