import React, {type ReactNode} from 'react';
import Layout from '@theme-original/DocItem/Layout';
import type LayoutType from '@theme/DocItem/Layout';
import type {Props} from '@theme/DocItem/Layout';
import type {WrapperProps} from '@docusaurus/types';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import SchemaInjector from '../../../components/SchemaInjector/SchemaInjector';
import { buildArticleSchema, buildProductSchema } from '@site/src/utils/schema-helpers';

export default function LayoutWrapper(props: Props): ReactNode {
  const { metadata, frontMatter } = useDoc();

  const title = metadata.title as string;
  const description = (frontMatter.description || metadata.description) as string | undefined;
  const url = metadata.permalink as string;
  const datePublished = ((frontMatter as any).date || (metadata as any).date || new Date().toISOString()) as string;
  const dateModified = metadata.lastUpdatedAt ? new Date(metadata.lastUpdatedAt).toISOString() : undefined;
  const authors = (frontMatter as any).authors as string[] | undefined;
  const image = frontMatter.image as string | undefined;

  const isMetricPage = metadata.permalink.includes('/docs/metrics-');

  const schema = isMetricPage 
    ? buildProductSchema({
        name: metadata.title,
        description: frontMatter.description || metadata.description,
        url: metadata.permalink
      })
    : buildArticleSchema({
      title,
      description,
      url,
      datePublished,
      dateModified,
      authors,
      image,
    });

  return (
    <>
      <SchemaInjector schema={schema} />
      <Layout {...props} />
    </>
  );
}