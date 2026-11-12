"use client";

import { useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import styles from "./JourneyGraph.module.scss";

type ClipPlayerProps = {
  src: string;
  label: string;
};

const clock = (seconds: number) => {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
};

// A playback bar for one recorded caller line.
const ClipPlayer: React.FC<ClipPlayerProps> = ({ src, label }) => {
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);

  const toggle = () => {
    const element = audio.current;
    if (!element) return;
    if (element.paused) void element.play().catch(() => undefined);
    else element.pause();
  };

  const seek = (event: React.MouseEvent<HTMLDivElement>) => {
    const element = audio.current;
    if (!element || !duration) return;
    const bar = event.currentTarget.getBoundingClientRect();
    element.currentTime = ((event.clientX - bar.left) / bar.width) * duration;
  };

  return (
    <div className={styles.player}>
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
      />
      <button
        type="button"
        onClick={toggle}
        aria-label={`${playing ? "Pause" : "Play"} ${label}`}
      >
        {playing ? (
          <Pause fill="currentColor" strokeWidth={0} />
        ) : (
          <Play fill="currentColor" strokeWidth={0} />
        )}
      </button>
      <div
        className={styles.seek}
        role="slider"
        aria-label="Position"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(time)}
        tabIndex={0}
        onClick={seek}
      >
        <div className={styles.track}>
          <div
            className={styles.played}
            style={{ width: duration ? `${(time / duration) * 100}%` : "0%" }}
          />
        </div>
      </div>
      <span className={styles.clock}>
        {clock(time)} / {clock(duration)}
      </span>
    </div>
  );
};

export default ClipPlayer;
