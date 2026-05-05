import type { ComponentProps } from "react";
import { PageLastUpdate } from "fumadocs-ui/layouts/notebook/page";
import type { Contributor } from "@/lib/contributors";
import CloudPlatformCallout from "@/src/components/CloudPlatformCallout";
import DiscordButton from "@/src/components/DiscordButton";
import GithubCtaButton from "@/src/components/GithubCtaButton";
import PageContributors from "@/src/components/PageContributors";
import styles from "./TocFooter.module.scss";

type Props = {
  contributors: Contributor[];
  lastModified?: ComponentProps<typeof PageLastUpdate>["date"];
};

const TocFooter: React.FC<Props> = ({ contributors, lastModified }) => {
  const hasMeta = contributors.length > 0 || Boolean(lastModified);

  return (
    <aside
      data-toc-full-bleed
      className={styles.footer}
      aria-label="Page metadata and community links"
    >
      <div className={styles.cloudSection}>
        <CloudPlatformCallout />
      </div>

      {hasMeta ? (
        <div className={styles.meta}>
          {contributors.length > 0 ? (
            <PageContributors contributors={contributors} />
          ) : null}
          {lastModified ? (
            <PageLastUpdate
              date={lastModified}
              className={styles.lastUpdated}
            />
          ) : null}
        </div>
      ) : null}

      <div className={styles.community}>
        <GithubCtaButton tone="secondary" alwaysCallout />
        <DiscordButton />
      </div>
    </aside>
  );
};


export default TocFooter;
