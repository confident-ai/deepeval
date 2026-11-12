"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, Zap } from "lucide-react";
import type { BenchmarkJourney } from "@/lib/benchmarks/types";
import styles from "./JourneyRecordings.module.scss";

type JourneyRecordingsProps = {
  journey: BenchmarkJourney;
};

type Clip = {
  key: string;
  src: string;
  text: string;
  // Set on a clip that is not the caller's own line: an aside, another voice.
  aside?: string;
};

// Every recording in a journey, in the order the caller could say them.
const clipsOf = (journey: BenchmarkJourney): Clip[] => [
  ...journey.lines.flatMap((line) => [
    ...line.events
      .filter((event) => event.audio)
      .map((event, index) => ({
        key: `${line.id}-event-${index}`,
        src: event.audio as string,
        text: event.label,
        aside: event.type,
      })),
    { key: line.id, src: line.audio, text: line.say },
  ]),
  ...journey.reactions.map((reaction) => ({
    key: `reaction-${reaction.id}`,
    src: reaction.audio,
    text: reaction.say,
  })),
];

const JourneyRecordings: React.FC<JourneyRecordingsProps> = ({ journey }) => {
  const clips = clipsOf(journey);
  const audio = useRef<HTMLAudioElement>(null);
  const rows = useRef<(HTMLLIElement | null)[]>([]);
  const [index, setIndex] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  // Whether the next clip should start on its own when this one ends.
  const [through, setThrough] = useState(false);

  useEffect(() => {
    if (index === null) return;
    const element = audio.current;
    if (!element) return;
    element.src = clips[index].src;
    void element.play().catch(() => undefined);
    rows.current[index]?.scrollIntoView({ block: "nearest" });
    // `clips` is rebuilt each render from the same journey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index]);

  const playAll = () => {
    if (through) {
      setThrough(false);
      audio.current?.pause();
      return;
    }
    setThrough(true);
    if (index === null || index === clips.length - 1) setIndex(0);
    else void audio.current?.play().catch(() => undefined);
  };

  const playOne = (clip: number) => {
    if (clip === index && playing) {
      audio.current?.pause();
      return;
    }
    setThrough(false);
    if (clip === index) void audio.current?.play().catch(() => undefined);
    else setIndex(clip);
  };

  const onEnded = () => {
    setPlaying(false);
    if (!through || index === null) return;
    if (index < clips.length - 1) setIndex(index + 1);
    else setThrough(false);
  };

  return (
    <aside className={styles.panel}>
      <audio
        ref={audio}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={onEnded}
      />
      <div className={styles.head}>
        <span>
          Recordings
          <small>{clips.length} clips</small>
        </span>
        <button type="button" className={styles.all} onClick={playAll}>
          {through ? (
            <Pause fill="currentColor" strokeWidth={0} />
          ) : (
            <Play fill="currentColor" strokeWidth={0} />
          )}
          {through ? "Pause" : "Play all"}
        </button>
      </div>
      <div className={styles.scroll}>
        <ol className={styles.list}>
          {clips.map((clip, position) => (
            <li
              key={clip.key}
              ref={(row) => {
                rows.current[position] = row;
              }}
              className={styles.row}
              data-active={position === index}
              data-aside={Boolean(clip.aside)}
            >
              <button
                type="button"
                onClick={() => playOne(position)}
                aria-label={`${position === index && playing ? "Pause" : "Play"} ${clip.text}`}
              >
                {position === index && playing ? (
                  <Pause fill="currentColor" strokeWidth={0} />
                ) : (
                  <Play fill="currentColor" strokeWidth={0} />
                )}
              </button>
              <span>
                {clip.aside && <Zap />}
                {clip.text}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </aside>
  );
};

export default JourneyRecordings;
