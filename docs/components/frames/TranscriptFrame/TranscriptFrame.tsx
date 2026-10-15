import type * as React from "react";
import { Check, X } from "lucide-react";
import FrameShell, { markColor } from "@/components/frames/FrameShell/FrameShell";
import type { ChartColor } from "@/lib/charts/schema";
import { TRANSCRIPT_KIND } from "@/lib/frames/kinds";
import { roleLabel, type TranscriptData, type TranscriptTurn } from "@/lib/frames/transcript";
import { renderHighlights } from "./_text";
import styles from "./styles.module.scss";

interface TranscriptFrameProps {
  id: string;
  transcript?: TranscriptData;

  type: string;
  title: string;

  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

const ROLE_COLORS: Record<TranscriptTurn["role"], ChartColor> = {
  user: "teal",
  assistant: "violet",
  system: "muted",
  tool: "ember",
};

const Verdict = ({ verdict }: { verdict?: "pass" | "fail" }): React.ReactElement | null =>
  verdict ? (
    <span
      className={styles.verdict}
      style={markColor(verdict === "pass" ? "teal" : "pink")}
      aria-label={verdict === "pass" ? "kept" : "rejected"}
    >
      {verdict === "pass" ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}
    </span>
  ) : null;

const Turn = ({ turn, speaker, tag }: { turn: TranscriptTurn; speaker?: string; tag?: string }): React.ReactElement => (
  <div className={styles.turn} style={markColor(ROLE_COLORS[turn.role])} data-role={turn.role}>
    <div className={styles.turnHead}>
      <span className={styles.role}>{roleLabel(turn, speaker)}</span>
      {tag ? <span className={styles.inlineTag}>{tag}</span> : null}
      <Verdict verdict={turn.verdict} />
    </div>
    <div className={styles.text}>{renderHighlights(turn.text, styles.highlight)}</div>
  </div>
);

/** A mock playground: each prompt in a field, each reply in an output panel. */
const Chat = ({ data }: { data: TranscriptData }): React.ReactElement => (
  <div className={styles.chat}>
    {data.turns.map((turn, i) => (
      <div key={i} className={turn.role === "user" ? styles.prompt : styles.reply} data-role={turn.role}>
        <div className={styles.turnHead}>
          <span className={styles.role}>{turn.role === "user" ? (turn.name ?? "Prompt") : roleLabel(turn, data.speaker)}</span>
          <Verdict verdict={turn.verdict} />
        </div>
        <div className={styles.text}>{renderHighlights(turn.text, styles.highlight)}</div>
      </div>
    ))}
  </div>
);

/** Turns stacked in order, with brackets over ranges and pointers at single turns. */
const Turns = ({ data }: { data: TranscriptData }): React.ReactElement => {
  const pointerAt = new Map(data.pointers.map(p => [p.turn, p.label]));
  return (
    <div className={styles.turns} style={{ gridTemplateRows: `repeat(${data.turns.length}, auto)` }}>
      {data.ranges.map((r, i) => (
        <div key={`r${i}`} className={styles.range} style={{ gridRow: `${r.from + 1} / ${r.to + 2}` }}>
          <span className={styles.rangeLabel}>{r.label}</span>
          <span className={styles.bracket} aria-hidden="true" />
        </div>
      ))}
      {data.turns.map((turn, i) => (
        <div key={`t${i}`} className={styles.turnCell} style={{ gridRow: i + 1 }}>
          <Turn turn={turn} speaker={data.speaker} tag={pointerAt.get(i)} />
        </div>
      ))}
      {data.pointers.map((p, i) => (
        <div key={`p${i}`} className={styles.pointer} style={{ gridRow: p.turn + 1 }}>
          <span className={styles.pointerArrow} aria-hidden="true" />
          {p.label}
        </div>
      ))}
    </div>
  );
};

const Conversation = ({ turns, speaker, title }: { turns: TranscriptTurn[]; speaker?: string; title?: string }): React.ReactElement => (
  <div className={styles.conversation}>
    {title ? <div className={styles.panelTitle}>{title}</div> : null}
    {turns.map((turn, i) => (
      <div key={i} className={styles.line}>
        <span className={styles.lineRole} style={markColor(ROLE_COLORS[turn.role])}>
          {roleLabel(turn, speaker)}
        </span>
        <span className={styles.text}>{renderHighlights(turn.text, styles.highlight)}</span>
      </div>
    ))}
  </div>
);

/** Conversation (or candidate panels) → judge prompt → verdict. */
const Judge = ({ data }: { data: TranscriptData }): React.ReactElement => {
  const judge = data.judge!;
  return (
    <div className={styles.judge}>
      <div className={styles.judgeInput}>
        {data.turns.length ? <Conversation turns={data.turns} speaker={data.speaker} /> : null}
        {data.panels.map((panel, i) => (
          <Conversation key={i} turns={panel.turns} speaker={data.speaker} title={panel.title} />
        ))}
      </div>
      <span className={styles.flowArrow} aria-hidden="true" />
      <div className={styles.judgeBox} style={markColor("yellow")}>
        <div className={styles.role}>{judge.label}</div>
        <div className={styles.text}>“{judge.prompt}”</div>
      </div>
      <span className={styles.flowArrow} aria-hidden="true" />
      <div className={styles.verdictBox}>
        <div className={styles.role}>{judge.verdictLabel}</div>
        <div className={styles.score} style={markColor("teal")}>
          {judge.verdict}
        </div>
      </div>
    </div>
  );
};

/** Panels side by side, each a titled column of turns. */
const Compare = ({ data }: { data: TranscriptData }): React.ReactElement => (
  <div className={styles.compare}>
    {data.panels.map((panel, i) => (
      <div key={i} className={styles.panel}>
        <div className={styles.panelTitle}>{panel.title}</div>
        {panel.turns.map((turn, j) => (
          <Turn key={j} turn={turn} speaker={data.speaker} />
        ))}
      </div>
    ))}
  </div>
);

const BODIES: Record<string, React.FC<{ data: TranscriptData }>> = { chat: Chat, turns: Turns, judge: Judge, compare: Compare };

const TranscriptFrame: React.FC<TranscriptFrameProps> = ({ id, transcript, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !transcript
    ? ["unknown or invalid data"]
    : !TRANSCRIPT_KIND.types.includes(type)
      ? [`unknown type "${type}"`]
      : TRANSCRIPT_KIND.typeProblems(type, transcript);
  if (!transcript || problems.length) {
    console.warn(`[TranscriptFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }
  const Body = BODIES[type];
  return (
    <FrameShell
      className={styles.frame}
      data={{ "data-transcript": id, "data-transcript-type": type }}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      note={note}
      content="text"
    >
      <Body data={transcript} />
    </FrameShell>
  );
};

/** Bind MDX to this article's transcripts.json; authors only supply the local ID. */
export const createTranscriptFrame = (data: Record<string, TranscriptData>) =>
  function ArticleTranscriptFrame(props: Omit<TranscriptFrameProps, "transcript">) {
    return <TranscriptFrame {...props} transcript={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default TranscriptFrame;
