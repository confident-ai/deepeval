import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        // RFC 8288 Link headers on the homepage so agents can discover
        // the docs, the llms.txt index, and the API catalog without
        // parsing HTML. (`Vary: Accept` for the negotiated routes is set in
        // `proxy.ts`, which can append to Next's own `Vary` instead of
        // replacing it.)
        source: '/',
        headers: [
          {
            key: 'Link',
            value:
              '</.well-known/api-catalog>; rel="api-catalog", </docs>; rel="service-doc", </llms.txt>; rel="describedby"',
          },
        ],
      },
      {
        // Discovery manifests (ARD ai-catalog.json, agent-skills index)
        // are meant to be fetched cross-origin by agents and registries.
        source: '/.well-known/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
        ],
      },
    ];
  },
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
