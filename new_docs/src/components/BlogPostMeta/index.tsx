import React from "react";
import { getAuthor, type AuthorId } from "@/lib/authors";
import { getBlogCategory, type BlogCategoryId } from "@/lib/blog-categories";
import styles from "./BlogPostMeta.module.scss";

interface BlogPostMetaProps {
  authors: AuthorId[];
  date: Date | string;
  category?: BlogCategoryId;
}

function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function isoDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toISOString().slice(0, 10);
}

const BlogPostMeta = ({ authors, date, category }: BlogPostMetaProps) => {
  const resolved = authors.map((id) => ({ id, ...getAuthor(id) }));
  const resolvedCategory = category ? getBlogCategory(category) : null;
  const CategoryIcon = resolvedCategory?.icon;

  return (
    <div className={styles.meta}>
      <div className={styles.avatars} aria-hidden="true">
        {resolved.map((author) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={author.id}
            src={author.imageUrl}
            alt=""
            className={styles.avatar}
          />
        ))}
      </div>
      <div className={styles.text}>
        <div className={styles.names}>
          {resolved.map((author, i) => (
            <React.Fragment key={author.id}>
              {i > 0 ? <span className={styles.separator}>, </span> : null}
              <a
                href={author.url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.name}
              >
                {author.name}
              </a>
            </React.Fragment>
          ))}
        </div>
        <time dateTime={isoDate(date)} className={styles.date}>
          {formatDate(date)}
        </time>
      </div>
      {resolvedCategory && CategoryIcon ? (
        <span className={styles.category}>
          <CategoryIcon className={styles.categoryIcon} aria-hidden="true" />
          <span>{resolvedCategory.label}</span>
        </span>
      ) : null}
    </div>
  );
};

export default BlogPostMeta;
