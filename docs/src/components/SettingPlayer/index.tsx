"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";

type Option = {
  /** What the pill shows, e.g. `Scottish` or `frequent`. */
  label: string;
  /** The setting as it is written in code, e.g. `VoiceSettings(accent="Scottish")`. */
  code: string;
  /** The recording of the caller with this setting. */
  src: string;
};

type Setting = {
  /** The parameter name, e.g. `accent`. */
  name: string;
  options: Option[];
};

interface SettingPlayerProps {
  settings: Setting[];
}

function clock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * A playback bar for recordings of the simulated caller, with a pill inside
 * it for each value of the setting. Picking a pill plays that recording; a
 * lone pill still names what the recording is.
 */
const SettingPlayer: React.FC<SettingPlayerProps> = ({ settings }) => {
  const [selected, setSelected] = useState(0);
  const [picked, setPicked] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);
  const options = settings.flatMap((setting) => setting.options);
  const option = options[selected];

  useEffect(() => {
    setTime(0);
    setDuration(0);
    // Only after a choice: a page should never start talking on load.
    if (picked) void audio.current?.play().catch(() => undefined);
  }, [selected, picked]);

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
    <div className="not-prose my-3 flex h-10 items-center gap-3 rounded-full bg-fd-muted pl-2 pr-1.5">
      <audio
        ref={audio}
        key={option.src}
        src={option.src}
        preload="none"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
      />
      <button
        type="button"
        onClick={toggle}
        aria-label={playing ? "Pause" : "Play"}
        className="flex size-7 shrink-0 items-center justify-center rounded-full text-fd-foreground transition-colors hover:bg-fd-border"
      >
        {playing ? (
          <Pause className="size-3.5" fill="currentColor" strokeWidth={0} />
        ) : (
          <Play className="size-3.5 translate-x-px" fill="currentColor" strokeWidth={0} />
        )}
      </button>
      <span className="shrink-0 text-xs tabular-nums text-fd-foreground">
        {clock(time)} / {clock(duration)}
      </span>
      <div
        role="slider"
        aria-label="Position"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(time)}
        tabIndex={0}
        onClick={seek}
        className="flex h-full min-w-12 flex-1 cursor-pointer items-center"
      >
        <div className="h-1 w-full overflow-hidden rounded-full bg-fd-border">
          <div
            className="h-full rounded-full bg-fd-foreground"
            style={{ width: duration ? `${(time / duration) * 100}%` : "0%" }}
          />
        </div>
      </div>
      {options.length > 0 && (
        <div className="flex min-w-0 items-center gap-0.5 overflow-x-auto rounded-full bg-fd-background p-0.5">
          {options.map((item, index) => (
            <button
              key={item.label}
              type="button"
              title={item.code}
              aria-pressed={index === selected}
              onClick={() => {
                setSelected(index);
                setPicked(true);
              }}
              className={
                "whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium leading-none transition-colors " +
                (index === selected
                  ? "bg-fd-foreground text-fd-background"
                  : "text-fd-muted-foreground hover:text-fd-foreground")
              }
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default SettingPlayer;
