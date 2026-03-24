const BASE_URL = "https://deepeval.com";

export interface ArticleSchemaProps {
  title: string
  url: string;
  description?: string;
  datePublished?: string;
  dateModified?: string;
  authors?: (string | undefined)[] | undefined;
  image?: string;
}

export interface BreadcrumbItem {
  name: string;
  url?: string;
}


export function buildWebSiteSchema(): object {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "DeepEval by Confident AI - The LLM Evaluation Framework", 
    url: BASE_URL,
  };
}

export function buildArticleSchema({
  title,
  description,
  url,
  datePublished,
  dateModified,
  authors,
  image,
}: ArticleSchemaProps): object {
  const authorSchema = authors && authors.length > 0 
    ? authors.map(name => ({
        "@type": "Person",
        name: name,
      }))
    : undefined;

  return {
    "@context": "https://schema.org",
    "@type": "TechArticle", 
    headline: title,
    ...(description ? { description } : {}),
    ...(image ? { image } : {}),
    ...(datePublished ? { datePublished } : {}),
    ...(dateModified ? { dateModified } : {}),
    mainEntityOfPage: { 
      "@type": "WebPage", 
      "@id": `${BASE_URL}${url}` 
    },
    ...(authorSchema 
        ? { author: authorSchema.length === 1 ? authorSchema[0] : authorSchema } 
        : {}
    ),
    publisher: {
      "@type": "Organization",
      name: "Confident AI Inc.",
      url: BASE_URL,
      logo: { 
        "@type": "ImageObject", 
        url: `${BASE_URL}/icons/DeepEval.svg` 
      },
    },
  };
}

export function buildBreadcrumbSchema(trail: BreadcrumbItem[]): object | null {
  if (!trail || trail.length === 0) return null;

  const items: object[] = [
    { "@type": "ListItem", position: 1, name: "Home", item: BASE_URL }
  ];

  let currentPosition = 2; 

  trail.forEach((crumb, i) => {
    const isLast = i === trail.length - 1;
    let itemUrl = crumb.url;

    if (itemUrl) {
      if (itemUrl.startsWith('/')) {
        itemUrl = `${BASE_URL}${itemUrl}`;
      } else if (!itemUrl.startsWith('http')) {
        itemUrl = `${BASE_URL}/${itemUrl}`;
      }
    }

    if (itemUrl || isLast) {
      items.push({
        "@type": "ListItem",
        position: currentPosition,
        name: crumb.name,
        ...(!isLast && itemUrl ? { item: itemUrl } : {}),
      });
      currentPosition++;
    }
  });

  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items,
  };
}