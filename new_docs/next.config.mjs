import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.ctfassets.net',
      },
      // Blog post hero / inline imagery — authored MDX references
      // `https://deepeval-docs.s3.us-east-1.amazonaws.com/...` directly
      // (e.g. `![alt](https://deepeval-docs.s3…png)`) and Next's MDX
      // pipeline lowers those to `next/image`, which rejects unknown
      // hosts. Allow the bucket explicitly rather than reaching for
      // `unoptimized: true`, so images still get optimized.
      {
        protocol: 'https',
        hostname: 'deepeval-docs.s3.us-east-1.amazonaws.com',
      },
    ],
  },
};

export default withMDX(config);
