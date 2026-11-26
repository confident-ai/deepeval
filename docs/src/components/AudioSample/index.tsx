"use client";

import React, { useEffect, useRef, useState } from "react";
import samples from "./samples.json";
import styles from "./AudioSample.module.scss";

interface Region {
  start: number;
  end: number;
  label: string;
}

interface Sample {
  duration: number;
  file: string;
  regions: Region[];
  lanes: { label: string; peaks: number[] }[];
}

interface AudioSampleProps {
  /** File name under `public/audio/voice-metrics/`, without `.mp3`. */
  name: string;
}

const seconds = (value: number) => `${value.toFixed(1)}s`;

const AudioSample: React.FC<AudioSampleProps> = ({ name }) => {
  const sample = (samples as Record<string, Sample>)[name];
  const audio = useRef<HTMLAudioElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const element = audio.current;
    if (!element) return;
    let frame = 0;
    const tick = () => {
      setTime(element.currentTime);
      frame = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      setPlaying(true);
      frame = requestAnimationFrame(tick);
    };
    const onStop = () => {
      setPlaying(false);
      cancelAnimationFrame(frame);
      setTime(element.currentTime);
    };
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onStop);
    element.addEventListener("ended", onStop);
    return () => {
      cancelAnimationFrame(frame);
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onStop);
      element.removeEventListener("ended", onStop);
    };
  }, []);

  if (!sample) return null;
  const { duration, regions, lanes } = sample;

  const seek = (to: number, play = true) => {
    const element = audio.current;
    if (!element) return;
    element.currentTime = Math.max(0, Math.min(duration, to));
    setTime(element.currentTime);
    if (play) void element.play();
  };

  const toggle = () => {
    const element = audio.current;
    if (!element) return;
    if (element.paused) {
      if (element.ended || element.currentTime >= duration - 0.05) element.currentTime = 0;
      void element.play();
    } else {
      element.pause();
    }
  };

  const onWaveClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    seek(((event.clientX - box.left) / box.width) * duration);
  };

  const progress = Math.min(1, time / duration);
  // A bar is flagged when any part of the time it covers overlaps a range,
  // so ranges shorter than one bar still show.
  const inRegion = (from: number, to: number) =>
    regions.some((region) => region.start < to && region.end > from);

  return (
    <figure className={styles.sample}>
      <audio ref={audio} preload="none" src={`/audio/voice-metrics/${sample.file}`} />
      <div className={styles.player}>
        <button
          type="button"
          className={styles.play}
          onClick={toggle}
          aria-label={playing ? "Pause example" : "Play example"}
        >
          {playing ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <rect x="3.5" y="2.5" width="3" height="11" rx="1" />
              <rect x="9.5" y="2.5" width="3" height="11" rx="1" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4.5 2.8v10.4a.8.8 0 0 0 1.2.7l8.4-5.2a.8.8 0 0 0 0-1.4L5.7 2.1a.8.8 0 0 0-1.2.7z" />
            </svg>
          )}
        </button>
        <div className={styles.lanes}>
          {lanes.length > 1 ? (
            <div className={styles.laneLabels}>
              {lanes.map((lane) => (
                <span key={lane.label}>{lane.label}</span>
              ))}
            </div>
          ) : null}
          <div
            className={styles.wave}
            onClick={onWaveClick}
            role="slider"
            aria-label="Seek"
            aria-valuemin={0}
            aria-valuemax={duration}
            aria-valuenow={Number(time.toFixed(1))}
          >
            {regions.map((region, index) => (
              <div
                key={index}
                className={styles.region}
                style={{
                  left: `${(region.start / duration) * 100}%`,
                  width: `${((region.end - region.start) / duration) * 100}%`,
                }}
              />
            ))}
            {lanes.map((lane) => (
              <div key={lane.label} className={styles.lane}>
                {lane.peaks.map((peak, index) => {
                  const from = (index / lane.peaks.length) * duration;
                  const to = ((index + 1) / lane.peaks.length) * duration;
                  const at = (from + to) / 2;
                  const classes = [styles.bar];
                  if (at <= time) classes.push(styles.played);
                  if (inRegion(from, to)) classes.push(styles.flagged);
                  return (
                    <span
                      key={index}
                      className={classes.join(" ")}
                      style={{ height: `${Math.max(6, peak * 100)}%` }}
                    />
                  );
                })}
              </div>
            ))}
            <div className={styles.playhead} style={{ left: `${progress * 100}%` }} />
          </div>
        </div>
        <span className={styles.time}>
          {seconds(time)} / {seconds(duration)}
        </span>
      </div>
    </figure>
  );
};

export default AudioSample;
