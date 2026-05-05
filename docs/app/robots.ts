import type { MetadataRoute } from 'next';
import { siteUrl } from '@/lib/shared';

// Matches the Docusaurus default (allow all crawlers, no disallow list)
// and additionally advertises our sitemap so search engines don't have
// to discover it blind. `host` is the canonical hostname.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: '*', allow: '/' }],
    sitemap: `${siteUrl}/sitemap.xml`,
    host: siteUrl,
  };
}
