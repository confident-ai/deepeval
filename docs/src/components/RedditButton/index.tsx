import type { ReactNode } from "react";
import Link from "next/link";
import { redditUrl } from "@/lib/shared";
import styles from "./RedditButton.module.scss";

// Inlined Reddit mark — lucide dropped brand icons in v0.475+.
export const RedditMark: React.FC<React.SVGProps<SVGSVGElement>> = (props) => {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12c-.69 0-1.25.56-1.25 1.25 0 .69.56 1.25 1.25 1.25.69 0 1.25-.56 1.25-1.25 0-.69-.56-1.25-1.25-1.25zm5.5 0c-.69 0-1.25.56-1.25 1.25 0 .69.56 1.25 1.25 1.25.69 0 1.25-.56 1.25-1.25 0-.69-.56-1.25-1.25-1.25zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z" />
    </svg>
  );
};

type RedditButtonProps = {
  label?: ReactNode;
  layout?: "full" | "inline" | "icon";
};

const RedditButton: React.FC<RedditButtonProps> = ({
  label = "Join Subreddit",
  layout = "full",
}) => {
  const iconOnly = layout === "icon";

  return (
    <Link
      href={redditUrl}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.root}
      data-layout={layout}
      aria-label={
        iconOnly || typeof label !== "string"
          ? "Join our Reddit community"
          : label
      }
      data-callout
      data-button
    >
      <RedditMark />
      {iconOnly ? null : label}
    </Link>
  );
};


export default RedditButton;
