import type * as React from "react";
import { markColor } from "@/components/frames/FrameShell/FrameShell";
import { iconTint, resolveIcon } from "@/lib/frames/board-icons";
import type { SceneIcon } from "./_scene";
import styles from "./styles.module.scss";

export const BoardIcon = ({ icon }: { icon: SceneIcon }): React.ReactElement | null => {
  const resolved = resolveIcon(icon.ref);
  if (!resolved) return null;
  const { x, y, w, h } = icon;
  const tint = iconTint(icon.ref);
  const tinted = tint ? { className: styles.tinted, style: markColor(tint) } : {};
  if (resolved.source === "brand") {
    return (
      <svg
        className={[styles.brand, tinted.className].filter(Boolean).join(" ")}
        x={x}
        y={y}
        width={w}
        height={h}
        viewBox="0 0 24 24"
        style={tinted.style ?? (resolved.color ? { fill: resolved.color } : undefined)}
      >
        <path d={resolved.path} />
      </svg>
    );
  }
  if (resolved.source === "icon") {
    const { Component } = resolved;
    return (
      <Component
        className={[styles.lineIcon, tinted.className].filter(Boolean).join(" ")}
        style={tinted.style}
        x={x}
        y={y}
        width={w}
        height={h}
        strokeWidth={1.75}
        aria-hidden="true"
      />
    );
  }
  return <image href={resolved.src} x={x} y={y} width={w} height={h} preserveAspectRatio="xMidYMid meet" />;
};
