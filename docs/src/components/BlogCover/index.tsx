import React from "react";
import styles from "./BlogCover.module.scss";

interface BlogCoverProps {
  src: string;
  alt: string;
}

/** A post's `cover.*`, shown under the byline. */
const BlogCover: React.FC<BlogCoverProps> = ({ src, alt }) => (
  <figure className={styles.cover}>
    <img src={src} alt={alt} />
  </figure>
);

export default BlogCover;
