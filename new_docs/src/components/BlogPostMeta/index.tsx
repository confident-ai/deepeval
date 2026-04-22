import { getAuthor, type AuthorId } from "@/lib/authors";
import { getBlogCategory, type BlogCategoryId } from "@/lib/blog-categories";
import styles from "./BlogPostMeta.module.scss";

interface BlogPostMetaProps {
  authors: AuthorId[];
  category?: BlogCategoryId;
}

const BlogPostMeta = ({ authors, category }: BlogPostMetaProps) => {
  const resolved = authors.map((id) => ({ id, ...getAuthor(id) }));
  const resolvedCategory = category ? getBlogCategory(category) : null;
  const CategoryIcon = resolvedCategory?.icon;

  // Single author keeps the compact inline row (avatar + name side by
  // side). With 2+ authors the overlapping-avatar + comma-joined names
  // starts reading as one blended unit, so we break each author onto
  // their own row — cleaner attribution for co-authored posts.
  return (
    <div className={styles.meta}>
      <ul className={styles.authorList}>
        {resolved.map((author) => (
          <li key={author.id} className={styles.author}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={author.imageUrl}
              alt=""
              className={styles.avatar}
              aria-hidden="true"
            />
            <a
              href={author.url}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.name}
            >
              {author.name}
            </a>
          </li>
        ))}
      </ul>
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
