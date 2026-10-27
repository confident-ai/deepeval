import React from "react";
import { resolveMedia } from "@/lib/media/library.mjs";
import styles from "./ImageDisplayer.module.scss";

interface ImageDisplayerProps {
  /** A media library id, e.g. `diagrams/in-house/evaluation/llm-test-case`. */
  id: string;
  /** Overrides the library's alt text. */
  alt?: string;
  width?: string | number;
  caption?: React.ReactNode;
}

const ImageDisplayer: React.FC<ImageDisplayerProps> = ({ id, alt, width, caption }) => {
  const image = resolveMedia(id);
  return (
    <figure className={styles.imageContainer}>
      <img
        src={image.src}
        alt={alt ?? image.alt}
        width={image.width}
        height={image.height}
        loading="lazy"
        style={width ? { width } : undefined}
      />
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  );
};

export default ImageDisplayer;
