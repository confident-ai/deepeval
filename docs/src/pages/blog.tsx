import React, { ReactNode, useMemo } from "react";
import styles from "./blog.module.scss";
import LayoutProvider from "@theme/Layout/Provider";
import Footer from "@theme/Footer";
import Navbar from "@theme/Navbar";
import Link from "@docusaurus/Link";
import SchemaInjector from "../components/SchemaInjector/SchemaInjector";
import { buildWebSiteSchema, buildBlogHomeSchema } from "@site/src/utils/schema-helpers";

interface Author {
  name: string;
  title: string;
  url: string;
  image_url: string;
}

interface BlogPost {
  title: string;
  description: string;
  slug: string;
  authors: string[];
  date: string;
}

const AUTHORS: Record<string, Author> = {
  penguine: {
    name: "Jeffrey Ip",
    title: "DeepEval Wizard",
    url: "https://github.com/penguine-ip",
    image_url: "https://github.com/penguine-ip.png"
  },
  kritinv: {
    name: "Kritin Vongthongsri",
    title: "DeepEval Guru",
    url: "https://github.com/kritinv",
    image_url: "https://github.com/kritinv.png"
  },
  cale: {
    name: "Cale",
    title: "DeepEval Scribe",
    url: "https://github.com/A-Vamshi",
    image_url: "https://github.com/A-Vamshi.png"
  }
};

const BLOG_POSTS: BlogPost[] = [
  {
    title: "Build and Evaluate a Multi-Turn Chatbot Using DeepEval",
    description: "Improve chatbot performance by evaluating conversation quality, memory, and custom metrics using DeepEval.",
    slug: "medical-chatbot-deepeval-guide",
    authors: ["cale"],
    date: "2025-06-24"
  },
  {
    title: "Evaluate a RAG-Based Contract Assistant with DeepEval",
    description: "Evaluate and deploy reliable RAG systems with DeepEval — test LLMs, detect hallucinations, and integrate into CI/CD workflows.",
    slug: "rag-contract-assistant-deepeval-guide",
    authors: ["cale", "penguine"],
    date: "2025-06-12"
  },
  {
    title: "How Cognee Used DeepEval to Validate Their AI Memory Research: A Case Study",
    description: "DeepEval is one of the top providers of G-Eval and in this article we'll share how to use it in the best possible way.",
    slug: "use-case-cognee-ai-memory",
    authors: ["penguine"],
    date: "2025-06-03"
  },
  {
    title: "Top 5 G-Eval Metric Use Cases in DeepEval",
    description: "DeepEval is one of the top providers of G-Eval and in this article we'll share how to use it in the best possible way.",
    slug: "top-5-geval-use-cases",
    authors: ["kritinv"],
    date: "2025-05-29"
  },
  {
    title: "All DeepEval Alternatives, Compared",
    description: "As the open-source LLM evaluation framework, DeepEval replaces a lot of alternatives that users might be considering.",
    slug: "deepeval-alternatives-compared",
    authors: ["penguine"],
    date: "2025-04-21"
  },
  {
    title: "DeepEval vs Arize",
    description: "DeepEval and Arize AI is similar in many ways, but DeepEval specializes in evaluation while Arize AI is mainly for observability.",
    slug: "deepeval-vs-arize",
    authors: ["kritinv"],
    date: "2025-04-21"
  },
  {
    title: "DeepEval vs Langfuse",
    description: "DeepEval and Langfuse solves different problems. While Langfuse is an entire platform for LLM observability, DeepEval focuses on modularized evaluation like Pytest.",
    slug: "deepeval-vs-langfuse",
    authors: ["kritinv"],
    date: "2025-03-31"
  },
  {
    title: "DeepEval vs Ragas",
    description: "As the open-source LLM evaluation framework, DeepEval offers everything Ragas offers but more including agentic and chatbot evaluations.",
    slug: "deepeval-vs-ragas",
    authors: ["penguine"],
    date: "2025-03-19"
  },
  {
    title: "DeepEval vs Trulens",
    description: "As the open-source LLM evaluation framework, DeepEval contains everything Trulens have, but also a lot more on top of it.",
    slug: "deepeval-vs-trulens",
    authors: ["penguine"],
    date: "2025-03-19"
  }
];

const BlogCard: React.FC<BlogPost & { authorsData: Record<string, Author> }> = ({
  title, description, date, authors, slug, authorsData
}) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric', month: 'long', day: 'numeric'
    });
  };

  return (
    <Link to={`/blog/${slug}`} className={styles.blogCard}>
      <div className={styles.blogCardHeader}>
        <h3 className={styles.title}>{title}</h3>
        <span className={styles.date}>{formatDate(date)}</span>
      </div>
      <p className={styles.description}>{description}</p>
      <div className={styles.meta}>
        <div className={styles.authors}>
          {authors.map((authorKey) => {
            const author = authorsData[authorKey];
            return author ? (
              <div key={authorKey} className={styles.authorItem}>
                <img src={author.image_url} alt={author.name} className={styles.authorImage} />
                <span className={styles.author}>{author.name}</span>
              </div>
            ) : <span key={authorKey}>{authorKey}</span>;
          })}
        </div>
        <div className={styles.readMore}>
          Read more
          <img src="/icons/right-arrow.svg" alt="arrow" />
        </div>
      </div>
    </Link>
  );
};

export default function BlogHome(): ReactNode {
  const sortedBlogs = useMemo(() =>
    [...BLOG_POSTS].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
  []);

  const websiteSchema = buildWebSiteSchema();
  const blogHomeSchema = useMemo(() => buildBlogHomeSchema(sortedBlogs), [sortedBlogs]);

  return (
    <LayoutProvider>
      <Navbar />
      <SchemaInjector schema={websiteSchema} />
      <SchemaInjector schema={blogHomeSchema} />

      <main className={styles.blogHomeContainer}>
        {/* Hero Section */}
        <section className={styles.blogHeroCard}>
          <div className={styles.content}>
            <span className={styles.tag}>DeepEval Blog</span>
            <h1>The LLM Evaluation Hub</h1>
            <p>
              Deep dives into LLM-as-a-judge, unit testing for RAG,
              and the latest research in AI quality assurance.
            </p>
            <a
              href="https://github.com/confident-ai/deepeval"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.githubButton}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              Star on GitHub
            </a>
          </div>
        </section>

        {/* Post Grid */}
        <section className={styles.blogContainer}>
          {sortedBlogs.length === 0 && (
            <div className={styles.noBlogsContainer}>
              <div className={styles.noBlogsText}>No blog posts available.</div>
            </div>
          )}
          {sortedBlogs.map((blog, index) => (
            <BlogCard
              key={index}
              {...blog}
              authorsData={AUTHORS}
            />
          ))}
        </section>
      </main>

      <Footer />
    </LayoutProvider>
  );
}