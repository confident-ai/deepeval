"use client";

import React from "react";
import { Compass } from "lucide-react";
import styles from "./VideoDisplayer.module.scss";
import { PrimaryButton } from "@site/src/components/Buttons";

const ENTERPRISE_HREF = "/enterprise";

interface VideoDisplayerProps {
  src: string;
  description: string;
  /** Currently unused — kept so callsites can keep passing it in case the CTA links back to Confident AI later. */
  confidentUrl?: string;
  label?: string;
  ctaText?: string;
}

const ConfidentLogoMark = () => (
  <svg
    width="22"
    height="22"
    viewBox="0 0 512 512"
    fill="currentColor"
    aria-label="Confident AI logo"
  >
    <path d="M0 130.247V381.73c0 2.904 2.458 5.191 5.356 5.005 59.688-3.821 123.532-24.779 168.612-66.136v-49.562h-65.679c-8.102 0-15.071-6.231-15.444-14.325-.398-8.635 6.484-15.749 15.021-15.749h66.103v-49.572c-44.828-41.139-108.672-62.34-168.614-66.149C2.457 125.058 0 127.344 0 130.247zM338.032 191.401v49.562h65.674c8.094 0 15.061 6.216 15.448 14.3.413 8.635-6.475 15.774-15.019 15.774h-66.103v49.572c44.828 41.139 108.673 62.34 168.614 66.149 2.897.184 5.354-2.102 5.354-5.005V130.27c0-2.904-2.458-5.19-5.356-5.005-59.843 3.829-123.609 24.831-168.612 66.136zM204.042 197.967v116.066h103.916V197.967H204.042z" />
  </svg>
);

const VideoDisplayer: React.FC<VideoDisplayerProps> = ({
  src,
  label,
  description,
  ctaText = "Explore Enterprise",
}) => {
  return (
    <div className={styles.videoContainer}>
      <video
        width="100%"
        muted
        autoPlay
        controls
        playsInline
        controlsList="nodownload"
      >
        <source src={src} type="video/mp4" />
      </video>
      <div className={styles.footer}>
        {/* <div className={styles.logoBadge}>
          <ConfidentLogoMark />
        </div> */}
        <div className={styles.copy}>
          {label && <p className={styles.heading}>{label}</p>}
          <p className={styles.description}>{description}</p>
        </div>
        <div className={styles.cta}>
          <PrimaryButton
            href={ENTERPRISE_HREF}
            startIcon={<Compass aria-hidden />}
            shortkey="E"
          >
            {ctaText}
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
};

export default VideoDisplayer;
