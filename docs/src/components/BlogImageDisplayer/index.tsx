import React from "react";
import styles from "./BlogImageDisplayer.module.scss";

interface BlogImageDisplayerProps {
  src: string;
  alt: string;
  caption?: string;
  cover?: boolean;
}

const BlogImageDisplayer = ({ 
  src, 
  caption, 
  alt, 
  cover 
}: BlogImageDisplayerProps) => {
  return (
    <div className={styles.imageContainer} style={{ marginTop: cover ? '1rem' : '' }}>
        <img className={styles.image} src={src} alt={alt} style={{ padding: cover ? '0' : '' }}/>
        {caption && <div className={styles.caption}>{caption}</div>}
    </div>
  );
}

export default BlogImageDisplayer;
