import { Cloud } from "lucide-react";
import { PrimaryButton } from "@/src/components/Buttons";
import { CONFIDENT_HOSTS_BY_NAME } from "@/src/utils/utm";
import styles from "./CloudPlatformCallout.module.scss";

const CloudPlatformCallout: React.FC = () => {
  return (
    <div className={styles.root}>
      <span className={styles.icon}>
        <Cloud aria-hidden="true" />
      </span>
      <span className={styles.content}>
        <span className={styles.title}>Collaborate in Confident Cloud</span>
        <span className={styles.body}>
          Review evals, traces, annotate, manage datasets, and version prompts.
        </span>
      </span>
      <span className={styles.cta}>
        <PrimaryButton
          href={CONFIDENT_HOSTS_BY_NAME.APP}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Explore Cloud Platform"
          data-utm-content="toc_cloud_platform"
        >
          Launch Platform
        </PrimaryButton>
      </span>
    </div>
  );
};


export default CloudPlatformCallout;
