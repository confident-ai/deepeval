import Link from "next/link";
import styles from "./ContributorDisplay.module.scss";

type Props = {
  href: string;
  avatarUrl: string;
  label: string;
  size?: "sm" | "md";
  title?: string;
  tooltip?: string;
};

const avatarSizes = {
  sm: 24,
  md: 32,
} as const;

const ContributorDisplay: React.FC<Props> = ({
  href,
  avatarUrl,
  label,
  size = "sm",
  title,
  tooltip,
}) => {
  const avatarSize = avatarSizes[size];

  return (
    <Link
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={label}
      title={title}
      className={styles.root}
      data-size={size}
      data-callout
      data-button
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={avatarUrl}
        alt=""
        className={styles.avatar}
        width={avatarSize}
        height={avatarSize}
        loading="lazy"
      />
      {tooltip ? (
        <span className={styles.tooltip} aria-hidden="true">
          {tooltip}
        </span>
      ) : null}
    </Link>
  );
};


export default ContributorDisplay;
