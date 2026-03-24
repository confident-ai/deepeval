import React, {type ReactNode} from 'react';
import BlogPostPage from '@theme-original/BlogPostPage';
import type BlogPostPageType from '@theme/BlogPostPage';
import type {WrapperProps} from '@docusaurus/types';
import type {Props} from '@theme/BlogPostPage';
import SchemaInjector from '../../components/SchemaInjector/SchemaInjector';
import { buildArticleSchema } from '@site/src/utils/schema-helpers';


export default function BlogPostPageWrapper(props: Props): ReactNode {
  const { content } = props;
  const { metadata, frontMatter } = content;

  const frontMatterAuthors = frontMatter.authors as string[] | undefined;
  const frontMatterDescription = frontMatter.description as string | undefined;
  const frontMatterImage = frontMatter.image as string | undefined;

  const authorNames = metadata.authors 
    ? metadata.authors.map(author => author.name) 
    : frontMatterAuthors;

  const articleSchema = buildArticleSchema({
    title: metadata.title,
    description: metadata.description || frontMatterDescription,
    url: metadata.permalink,
    datePublished: metadata.date,
    dateModified: metadata.date,
    authors: authorNames,
    image: frontMatterImage || undefined,
  });

  return (
    <>
      <SchemaInjector schema={articleSchema} />
      <BlogPostPage {...props} />
    </>
  );
}