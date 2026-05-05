import Link from "next/link";
import { getAuthor, type AuthorId } from "@/lib/authors";
import { getBlogCategory, type BlogCategoryId } from "@/lib/blog-categories";
import styles from "./BlogPostMeta.module.scss";

interface BlogPostMetaProps {
  authors: AuthorId[];
  category?: BlogCategoryId;
}

const BlogPostMeta: React.FC<BlogPostMetaProps> = ({ authors, category }) => {
  const resolved = authors.map((id) => ({ id, ...getAuthor(id) }));
  const [leadAuthor, ...coAuthors] = resolved;
  const resolvedCategory = category ? getBlogCategory(category) : null;
  const CategoryIcon = resolvedCategory?.icon;

  return (
    <div className={styles.meta}>
      <div className={styles.authorBlock}>
        <div className={styles.leadAuthor}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={leadAuthor.imageUrl}
            alt=""
            className={styles.avatar}
            aria-hidden="true"
          />
          <div className={styles.authorText}>
            <span className={styles.authorLabel}>First author</span>
            <Link
              href={leadAuthor.url}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.name}
            >
              {leadAuthor.name}
            </Link>
          </div>
        </div>

        {coAuthors.length > 0 ? (
          <ul className={styles.coAuthorList}>
            {coAuthors.map((author) => (
              <li key={author.id} className={styles.coAuthor}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={author.imageUrl}
                  alt=""
                  className={styles.coAuthorAvatar}
                  aria-hidden="true"
                />
                <div className={styles.authorText}>
                  <span className={styles.coAuthorLabel}>Co-author</span>
                  <Link
                    href={author.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.coAuthorName}
                  >
                    {author.name}
                  </Link>
                </div>
              </li>
            ))}
          </ul>
        ) : null}
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
