"use client";

import { useRef, useState } from "react";
import { Pause, Play } from "lucide-react";

interface PlayButtonProps {
  /** The recording to play. */
  src: string;
  /** What the recording is, for screen readers. */
  label: string;
}

/** A single play button for a recording, small enough for a table cell. */
const PlayButton: React.FC<PlayButtonProps> = ({ src, label }) => {
  const [playing, setPlaying] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);

  const toggle = () => {
    const element = audio.current;
    if (!element) return;
    if (element.paused) void element.play().catch(() => undefined);
    else element.pause();
  };

  return (
    <span className="not-prose inline-flex">
      <audio
        ref={audio}
        src={src}
        preload="none"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <button
        type="button"
        onClick={toggle}
        aria-label={`${playing ? "Pause" : "Play"} ${label}`}
        className="flex size-7 items-center justify-center rounded-full bg-fd-muted text-fd-foreground transition-colors hover:bg-fd-border"
      >
        {playing ? (
          <Pause className="size-3.5" fill="currentColor" strokeWidth={0} />
        ) : (
          <Play className="size-3.5 translate-x-px" fill="currentColor" strokeWidth={0} />
        )}
      </button>
    </span>
  );
};

export default PlayButton;
