import type * as React from "react";
import type { ChartColor } from "@/lib/charts/schema";
import styles from "./styles.module.scss";

export interface LegendEntry {
  id: string;
  label: string;
  color: ChartColor;
  /** Draw a line swatch instead of a square. */
  line?: boolean;
  dashed?: boolean;
}

interface FrameShellProps {
  /** Frame-specific class for the figure; its nested rules style the drawing. */
  className?: string;
  /** data-* attributes for the figure, e.g. { "data-chart": id }. */
  data?: Record<`data-${string}`, string>;
  title: string;
  description: string;
  kicker?: string;
  subtitle?: string;
  caption?: string;
  legend?: LegendEntry[];
  note?: string;
  /**
   * "image" for drawings (SVG): screen readers get the description instead.
   * "text" for frames made of real text: they read it, after the description.
   */
  content?: "image" | "text";
  children: React.ReactNode;
}

const Title = ({ text }: { text: string }): React.ReactElement => (
  <div className={styles.title}>
    {text.split(/(\*[^*]+\*)/).map((part, i) =>
      part.startsWith("*") && part.endsWith("*") ? <em key={i}>{part.slice(1, -1)}</em> : part
    )}
  </div>
);

// A caption may credit its source with a markdown link: [DeepEval](https://deepeval.com).
const Caption = ({ text }: { text: string }): React.ReactElement => (
  <figcaption className={styles.caption}>
    {text.split(/(\[[^\]]+\]\([^)\s]+\))/).map((part, i) => {
      const link = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      if (!link) return part;
      const external = /^https?:\/\//.test(link[2]) && !/^https?:\/\/(www\.)?deepeval\.com(\/|$)/.test(link[2]);
      return (
        <a key={i} href={link[2]} {...(external ? { target: "_blank", rel: "noopener" } : {})}>
          {link[1]}
        </a>
      );
    })}
  </figcaption>
);

export const markColor = (name: ChartColor) => ({ "--mark-color": `var(--chart-${name})` }) as React.CSSProperties;

const Legend = ({ entries, note }: { entries: LegendEntry[]; note?: string }): React.ReactElement | null => {
  if (!entries.length && !note) return null;
  return (
    <div className={styles.legend}>
      {entries.map(e => (
        <span key={e.id} className={styles.legendItem}>
          <span
            className={e.line ? styles.lineSwatch : styles.swatch}
            data-dashed={e.dashed || undefined}
            style={markColor(e.color)}
          />
          {e.label}
        </span>
      ))}
      {note ? <span className={styles.note}>{note}</span> : null}
    </div>
  );
};

/** The figure every frame shares: head, drawing, legend and caption. */
const FrameShell: React.FC<FrameShellProps> = ({
  className,
  data,
  title,
  description,
  kicker,
  subtitle,
  caption,
  legend = [],
  note,
  content = "image",
  children,
}) => (
  <figure className={className ? `${styles.frame} ${className}` : styles.frame} {...data}>
    <div className={styles.head}>
      {kicker ? <div className={styles.kicker}>{kicker}</div> : null}
      <Title text={title} />
      {subtitle ? <div className={styles.subtitle}>{subtitle}</div> : null}
    </div>
    {content === "image" ? (
      <div className={styles.drawing} role="img" aria-label={description} tabIndex={0}>
        {children}
      </div>
    ) : (
      <div className={styles.drawing}>
        <p className={styles.srOnly}>{description}</p>
        {children}
      </div>
    )}
    <Legend entries={legend} note={note} />
    {caption ? <Caption text={caption} /> : null}
  </figure>
);

export default FrameShell;
