import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import styles from "./HeroAnnouncement.module.scss";

type HeroAnnouncementProps = {
  href: string;
  label: string;
  children: ReactNode;
};

export const HeroAnnouncement: React.FC<HeroAnnouncementProps> = ({
  href,
  label,
  children,
}) => {
  return (
    <Link href={href} className={styles.root} aria-label={label} data-callout>
      <span className={styles.badge}>NEW</span>
      <span className={styles.content}>{children}</span>
      <span className={styles.icon} aria-hidden="true">
        <ArrowUpRight />
      </span>
    </Link>
  );
};

export default HeroAnnouncement;
