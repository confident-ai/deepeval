import type * as React from "react";
import { BOARD_SIZES, type BoardSize } from "@/lib/frames/board";

/**
 * Every size a board draws comes from one of four tiers. `text` and `line` are the font size and the
 * height of one line; `inline` is an icon in front of text of this tier, `icon` an icon standing on its
 * own (in a box, above a label); `gap` separates an inline icon from its text.
 */
export interface TierTokens {
  text: number;
  line: number;
  inline: number;
  icon: number;
  gap: number;
}

export const TIER: Record<BoardSize, TierTokens> = {
  xs: { text: 11, line: 16, inline: 14, icon: 20, gap: 4 },
  sm: { text: 12, line: 18, inline: 16, icon: 24, gap: 6 },
  md: { text: 14, line: 20, inline: 18, icon: 40, gap: 6 },
  lg: { text: 18, line: 26, inline: 22, icon: 64, gap: 8 },
};

/** What each part of a board is, and the tier it's drawn at unless the board sets a `size`. */
export const ROLE = {
  /** The board's heading. */
  heading: "lg",
  /** Box titles, tree rows, column headings, a bare pullout's heading. */
  title: "md",
  /** Cell and circle titles, notes, a circle pullout's card. */
  label: "sm",
  /** Lines of text inside a box. */
  body: "sm",
  /** Link and line labels, pills. */
  annotation: "xs",
  /** A bare pullout: the user, an input, an output. */
  actor: "md",
  /** Icons standing in a box. */
  icon: "sm",
} as const satisfies Record<string, BoardSize>;

export type Role = keyof typeof ROLE;

/** Roles drawn at medium weight; every other role is regular. */
const MEDIUM: ReadonlySet<Role> = new Set<Role>(["heading", "title", "actor"]);

export const weightOf = (role: Role): "medium" | undefined => (MEDIUM.has(role) ? "medium" : undefined);

/** An ellipsis ("⋮", "…") is drawn as three dots this size, readable whatever tier its text would be. */
export const ELLIPSIS = { r: 2.25, pitch: 8 };

/** Space between two inline icons. */
export const INLINE_GAP = 4;

/** The font sizes the stylesheet reads, so drawn text is always the size layout measured. */
export const sizeVars = Object.fromEntries(BOARD_SIZES.map(size => [`--board-text-${size}`, `${TIER[size].text}px`])) as React.CSSProperties;
