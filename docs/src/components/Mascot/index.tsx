"use client";

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type PointerEvent,
} from "react";
import {
  ACCESSORIES,
  BODY,
  BOWTIE,
  EARS,
  FACE_ROW,
  FACES,
  GRID,
  ICE,
  LEGS,
  NOTES,
  OVERLAYS,
  BOOK,
  VEIN,
  POSES,
  WALK_LEGS,
  type MascotAccessory,
  type MascotOverlay,
  type MascotExpression,
  type MascotPose,
  type Sprite,
} from "./_sprites";
import { pixelPaths } from "./_pixels";
import styles from "./Mascot.module.scss";

export type { MascotAccessory, MascotExpression, MascotOverlay, MascotPose };
export type MascotLook = "left" | "center" | "right";
export type MascotMotion = "jump" | "shake" | "dance" | "fly";
export type MascotEffect = "freeze" | "music" | "book" | "vein";

export interface MascotReaction {
  expression?: MascotExpression;
  pose?: MascotPose;
  look?: MascotLook;
  motion?: MascotMotion;
  effect?: MascotEffect;
}

export type MascotReactionPreset =
  | "smile"
  | "wave"
  | "cheer"
  | "jump"
  | "shake"
  | "dance"
  | "cry"
  | "freeze"
  | "angry"
  | "fly"
  | "read"
  | "wink";

type Reaction = MascotReaction | MascotReactionPreset;

const PRESETS: Record<MascotReactionPreset, MascotReaction> = {
  smile: { expression: "happy" },
  wave: { expression: "happy", pose: "wave" },
  cheer: { expression: "happy", pose: "cheer" },
  jump: { expression: "happy", motion: "jump" },
  shake: { expression: "surprised", motion: "shake" },
  dance: {
    expression: "happy",
    pose: "cheer",
    motion: "dance",
    effect: "music",
  },
  cry: { expression: "cry" },
  freeze: { expression: "surprised", effect: "freeze" },
  angry: { expression: "angry", effect: "vein" },
  fly: { expression: "happy", motion: "fly" },
  read: { expression: "closed", effect: "book" },
  wink: { expression: "wink", pose: "wave" },
};

const CLICK_MS = 1600;

const ICE_TINT = "#bfe6ff";

const PALETTE: Record<string, string> = {
  B: "var(--mascot-body)",
  S: "color-mix(in srgb, var(--mascot-body) 85%, black)",
  D: "color-mix(in srgb, var(--mascot-body) 65%, black)",
  E: "var(--mascot-ink)",
  M: "var(--mascot-ink)",
  P: "color-mix(in srgb, #ff7a9c 60%, var(--mascot-body))",
  U: "#6ec6ff",
  V: "#c9ced6",
  O: "#8a5a3c",
  N: "var(--mascot-body)",
  I: "#d9f1ff",
  i: "color-mix(in srgb, #bfe6ff 35%, transparent)",
  T: "var(--mascot-tie)",
  A: "var(--mascot-accessory)",
  a: "color-mix(in srgb, var(--mascot-accessory) 72%, black)",
  W: "#ffffff",
  Q: "#d6dbe1",
  Y: "#ffd84a",
  R: "#e5484d",
};

const LOOK_SHIFT: Record<MascotLook, number> = {
  left: -1,
  center: 0,
  right: 1,
};

const MOTION_CLASS: Record<MascotMotion, string> = {
  jump: styles.jump,
  shake: styles.shake,
  dance: styles.dance,
  fly: styles.fly,
};

export interface MascotProps {
  accessory?: MascotAccessory;
  overlay?: MascotOverlay;
  expression?: MascotExpression;
  look?: MascotLook;
  pose?: MascotPose;
  motion?: MascotMotion;
  ears?: boolean;
  hover?: Reaction;
  click?: Reaction | readonly Reaction[];
  onClick?: (event: MouseEvent<HTMLButtonElement>) => void;
  color?: string;
  tieColor?: string;
  accessoryColor?: string;
  inkColor?: string;
  animated?: boolean;
  animationDelay?: number;
  label?: string;
  className?: string;
}

export type Character = Omit<
  MascotProps,
  "look" | "pose" | "motion" | "animated" | "animationDelay"
> & {
  speed?: number;
};

const resolve = (reaction?: Reaction): MascotReaction | undefined =>
  typeof reaction === "string" ? PRESETS[reaction] : reaction;

const toPaths = (sprite: Sprite, row = 0, shift = 0) =>
  pixelPaths(sprite, PALETTE, row, shift);

const only = (face: Sprite, keys: RegExp) =>
  face.map((line) => line.replace(keys, "."));

const Face = ({
  expression,
  shift,
}: {
  expression: MascotExpression;
  shift: number;
}) => {
  const face = FACES[expression];
  return (
    <g>
      {toPaths(only(face, /[EU]/g), FACE_ROW, shift)}
      <g className={expression === "neutral" ? styles.eyes : undefined}>
        {toPaths(only(face, /[^E]/g), FACE_ROW, shift)}
      </g>
      <g className={styles.tears}>
        {toPaths(only(face, /[^U]/g), FACE_ROW, shift)}
      </g>
    </g>
  );
};

const Frames = ({
  frames,
  row = GRID - frames[0].length,
}: {
  frames: readonly Sprite[];
  row?: number;
}) => {
  return frames.length === 1 ? (
    <>{toPaths(frames[0], row)}</>
  ) : (
    <>
      <g className={styles.frameA}>{toPaths(frames[0], row)}</g>
      <g className={styles.frameB}>{toPaths(frames[1], row)}</g>
    </>
  );
};

const Mascot = ({
  accessory = "none",
  overlay,
  expression = "neutral",
  look = "center",
  pose = "idle",
  motion,
  ears = true,
  hover,
  click,
  onClick,
  color = "#b8a4ff",
  tieColor = "#161616",
  accessoryColor,
  inkColor = "#161616",
  animated = false,
  animationDelay = 0,
  label,
  className,
}: MascotProps) => {
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState<{
    reaction: MascotReaction;
    id: number;
  } | null>(null);
  const nextClick = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => () => clearTimeout(timer.current), []);

  const reaction = clicked?.reaction ?? (hovered ? resolve(hover) : undefined);
  const current = {
    expression: reaction?.expression ?? expression,
    pose: reaction?.pose ?? pose,
    look: reaction?.look ?? look,
  };
  const shift = LOOK_SHIFT[current.look];
  const activeMotion = reaction?.motion ?? motion;
  const frozen = reaction?.effect === "freeze";
  const clicks = click === undefined ? [] : [click].flat();
  const isButton = click !== undefined || onClick !== undefined;

  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    onClick?.(event);
    if (clicks.length === 0) return;
    const next = resolve(clicks[nextClick.current % clicks.length]);
    nextClick.current += 1;
    setClicked({ reaction: next ?? {}, id: nextClick.current });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setClicked(null), CLICK_MS);
  };

  const hoverHandlers = hover
    ? {
        onPointerEnter: (e: PointerEvent) => {
          if (e.pointerType !== "touch") setHovered(true);
        },
        onPointerLeave: () => setHovered(false),
        onFocus: () => setHovered(true),
        onBlur: () => setHovered(false),
      }
    : {};

  const svg = (
    <svg
      className={[
        styles.mascot,
        animated && styles.animated,
        reaction && styles.reacting,
        current.pose === "walk" && styles.walking,
        frozen && styles.frozen,
        !isButton && className,
      ]
        .filter(Boolean)
        .join(" ")}
      viewBox={`0 0 ${GRID} ${GRID}`}
      shapeRendering="crispEdges"
      role={label && !isButton ? "img" : undefined}
      aria-label={isButton ? undefined : label}
      aria-hidden={label && !isButton ? undefined : true}
      {...(isButton ? {} : hoverHandlers)}
      style={
        {
          "--mascot-body": frozen
            ? `color-mix(in srgb, ${color} 45%, ${ICE_TINT})`
            : color,
          "--mascot-tie": tieColor,
          "--mascot-accessory":
            accessoryColor ??
            "color-mix(in srgb, var(--mascot-body) 45%, black)",
          "--mascot-ink": inkColor,
          "--mascot-delay": `${animationDelay}s`,
        } as CSSProperties
      }
    >
      {label && !isButton && <title>{label}</title>}
      <g
        key={clicked?.id}
        className={activeMotion && MOTION_CLASS[activeMotion]}
      >
        {current.pose === "walk" ? (
          <g
            transform={`translate(${WALK_LEGS.x} ${WALK_LEGS.y}) scale(${WALK_LEGS.scale})`}
          >
            <Frames frames={WALK_LEGS.frames} row={0} />
          </g>
        ) : (
          toPaths(LEGS)
        )}
        <g className={styles.bob}>
          {toPaths(BODY)}
          {ears && toPaths(EARS)}
          <Frames frames={POSES[current.pose]} />
          <Face expression={current.expression} shift={shift} />
          {accessory !== "none" && (
            <g className={accessory === "copter" ? styles.spin : undefined}>
              <Frames frames={ACCESSORIES[accessory]} />
            </g>
          )}
          {overlay && (
            <Frames
              frames={[
                OVERLAYS[overlay][
                  accessory !== "none" &&
                  ACCESSORIES[accessory][0].length - GRID >= 2
                    ? "tall"
                    : "regular"
                ],
              ]}
            />
          )}
          <g
            transform={`translate(${BOWTIE.x} ${BOWTIE.y}) scale(${BOWTIE.scale})`}
          >
            {toPaths(BOWTIE.rows)}
          </g>
          {reaction?.effect === "book" && toPaths(BOOK)}
          {reaction?.effect === "vein" && (
            <g className={styles.vein}>{toPaths(VEIN.rows, VEIN.y, VEIN.x)}</g>
          )}
        </g>
      </g>
      {frozen && toPaths(ICE)}
      {reaction?.effect === "music" &&
        NOTES.map((note, i) => (
          <g key={i} className={styles.note}>
            {toPaths(note.rows, note.y, note.x)}
          </g>
        ))}
    </svg>
  );

  if (!isButton) return svg;

  return (
    <button
      type="button"
      className={[styles.button, className].filter(Boolean).join(" ")}
      aria-label={label ?? "Confident AI mascot"}
      onClick={handleClick}
      {...hoverHandlers}
    >
      {svg}
    </button>
  );
};

export default Mascot;
