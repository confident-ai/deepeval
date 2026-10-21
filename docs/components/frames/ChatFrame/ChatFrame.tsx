import type * as React from "react";
import { ArrowUp, Bot, Check, X } from "lucide-react";
import FrameShell from "@/components/frames/FrameShell/FrameShell";
import { renderHighlights } from "@/components/frames/TranscriptFrame/_text";
import type { ChatData, ChatMessage } from "@/lib/frames/chat";
import { CHAT_KIND } from "@/lib/frames/kinds";
import styles from "./styles.module.scss";

interface ChatFrameProps {
  id: string;
  chat?: ChatData;
  type: string;
  title: string;
  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

const Verdict = ({ verdict }: { verdict?: ChatMessage["verdict"] }): React.ReactElement | null =>
  verdict ? (
    <span className={styles.verdict} data-verdict={verdict} aria-label={verdict === "pass" ? "acceptable reply" : "unsafe reply"}>
      {verdict === "pass" ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}
    </span>
  ) : null;

const Text = ({ text }: { text: string }): React.ReactElement => <>{renderHighlights(text, styles.highlight)}</>;

/** Bubbles in a chat window: the user's on the right, the bot's on the left beside its avatar. */
const Chat = ({ data }: { data: ChatData }): React.ReactElement => (
  <div className={styles.window}>
    <div className={styles.windowHead}>
      <span className={styles.avatar} aria-hidden="true">
        <Bot size={14} />
      </span>
      {data.bot}
    </div>
    <div className={styles.messages}>
      {data.messages.map((m, i) =>
        m.from === "user" ? (
          <div key={i} className={styles.userBubble}>
            <Text text={m.text} />
          </div>
        ) : (
          <div key={i} className={styles.botRow}>
            <span className={styles.avatar} aria-hidden="true">
              <Bot size={14} />
            </span>
            <div className={styles.botBubble}>
              <Text text={m.text} />
            </div>
            <Verdict verdict={m.verdict} />
          </div>
        )
      )}
    </div>
    <div className={styles.composer} aria-hidden="true">
      <span>Message {data.bot}…</span>
      <span className={styles.send}>
        <ArrowUp size={12} strokeWidth={2.5} />
      </span>
    </div>
  </div>
);

/** A playground: each prompt in a field with a Generate button, each reply in a terminal-style output panel. */
const Playground = ({ data }: { data: ChatData }): React.ReactElement => (
  <div className={styles.playground}>
    {data.messages.map((m, i) =>
      m.from === "user" ? (
        <div key={i} className={styles.promptBlock}>
          <span className={styles.fieldLabel}>Prompt</span>
          <div className={styles.field}>
            <Text text={m.text} />
          </div>
          <span className={styles.generate} aria-hidden="true">
            Generate
          </span>
        </div>
      ) : (
        <div key={i} className={styles.outputBlock}>
          <div className={styles.outputHead}>
            <span className={styles.botName}>{data.bot}</span>
            <Verdict verdict={m.verdict} />
          </div>
          <div className={styles.output}>
            <Text text={m.text} />
          </div>
        </div>
      )
    )}
  </div>
);

const ChatFrame: React.FC<ChatFrameProps> = ({ id, chat, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !chat
    ? ["unknown or invalid data"]
    : !CHAT_KIND.types.includes(type)
      ? [`unknown type "${type}"`]
      : CHAT_KIND.typeProblems(type, chat);
  if (!chat || problems.length) {
    console.warn(`[ChatFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }
  return (
    <FrameShell
      className={styles.frame}
      data={{ "data-chat": id, "data-chat-type": type }}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      note={note}
      content="text"
    >
      {type === "playground" ? <Playground data={chat} /> : <Chat data={chat} />}
    </FrameShell>
  );
};

export const createChatFrame = (data: Record<string, ChatData>) =>
  function ArticleChatFrame(props: Omit<ChatFrameProps, "chat">) {
    return <ChatFrame {...props} chat={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default ChatFrame;
