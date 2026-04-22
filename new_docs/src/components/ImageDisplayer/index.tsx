import React from "react";
import styles from "./ImageDisplayer.module.scss";

interface ImageDisplayerProps {
  src: string;
  alt?: string;
  width?: string | number;
  caption?: React.ReactNode;
}

const ImageDisplayer = ({ src, alt, width, caption }: ImageDisplayerProps) => {
  return (
    <figure className={styles.imageContainer}>
      <img src={src} alt={alt ?? ""} style={width ? { width } : undefined} />
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  );
};

export default ImageDisplayer;
