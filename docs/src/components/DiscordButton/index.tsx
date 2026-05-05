import type { ReactNode } from "react";
import Link from "next/link";
import { discordUrl } from "@/lib/shared";
import styles from "./DiscordButton.module.scss";

/**
 * Inlined Discord "Clyde" wordless mark.
 *
 * Why not `lucide-react`: lucide removed all brand icons in v0.475+
 * (Discord, GitHub, Twitter, …) because brand marks are trademarks.
 *
 * Path data is the canonical Discord mark from their brand kit
 * (https://discord.com/branding). `fill="currentColor"` + the
 * module's `fill: currentColor` rule lets the button's `color` token
 * paint the glyph in one place, so a future theme swap only touches
 * the container.
 */
export const DiscordMark: React.FC<React.SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M20.317 4.369A19.79 19.79 0 0 0 16.885 3.3a.074.074 0 0 0-.079.037c-.34.6-.719 1.382-.984 1.995a18.307 18.307 0 0 0-5.487 0A12.72 12.72 0 0 0 9.335 3.337.077.077 0 0 0 9.256 3.3a19.735 19.735 0 0 0-3.432 1.069.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.08.08 0 0 0 .031.055 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.075.075 0 0 0-.041-.104 13.098 13.098 0 0 1-1.872-.892.075.075 0 0 1-.007-.125c.126-.094.252-.192.372-.29a.075.075 0 0 1 .078-.01c3.927 1.793 8.18 1.793 12.061 0a.075.075 0 0 1 .079.009c.12.098.245.196.372.291a.075.075 0 0 1-.006.125c-.598.349-1.22.645-1.873.891a.075.075 0 0 0-.041.105 14.42 14.42 0 0 0 1.226 1.994.076.076 0 0 0 .084.028 19.84 19.84 0 0 0 6.003-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.06.06 0 0 0-.031-.029zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.955 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
    </svg>
  );
};

/**
 * "Join Community" CTA that links to the DeepEval Discord in Blurple.
 * Pure link semantics — no JS on the client,
 * and the URL comes from `lib/shared.ts` so the rest of the site
 * (Kapa disclaimer copy, footers, etc.) stays consistent.
 */
type DiscordButtonProps = {
  label?: ReactNode;
  layout?: "full" | "inline";
};

const DiscordButton: React.FC<DiscordButtonProps> = ({
  label = "Join Community",
  layout = "full",
}) => {
  return (
    <Link
      href={discordUrl}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.root}
      data-layout={layout}
      aria-label={typeof label === "string" ? label : "Join our Discord community"}
      data-callout
      data-button
    >
      <DiscordMark />
      {label}
    </Link>
  );
};


export default DiscordButton;
