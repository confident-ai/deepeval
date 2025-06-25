import React from "react";
import styles from "./VideoDisplayer.module.css";

const VideoDisplayer = ({ src, confidentUrl, label }) => {
  return (
    <div className={styles.videoContainer}>
      <video width="100%" muted controls playsInline controlsList="nodownload">
        <source
          src={src}
          type="video/mp4"
        />
      </video>
      <div className={styles.overlay}>
        <div className={styles.playButton} onClick={() => window.open("https://documentation.confident-ai.com" + confidentUrl, '_blank')}>
          {label}
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            width="16" 
            height="16" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
            <polyline points="15 3 21 3 21 9"></polyline>
            <line x1="10" y1="14" x2="21" y2="3"></line>
          </svg>
        </div>
      </div>
    </div>
  );
}

export default VideoDisplayer;
