// src/components/CopyMarkdownButton.js
import React from 'react';
import { useLocation } from '@docusaurus/router';

const GITHUB_REPO = "confident-ai/deepeval";
const GITHUB_BRANCH = "main";

const CopyMarkdownButton = () => {
  const location = useLocation();

  const copyMarkdown = async () => {
    // Infer the MDX file path from the URL
    let path = location.pathname.replace(/^\/docs\//, '').replace(/\/$/, '');
    if (!path) path = 'index'; // fallback for root docs page
    const mdxFile = `${path}.mdx`;

    // Construct the raw GitHub URL
    const rawUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/docs/docs/${mdxFile}`;
    // Construct the web GitHub URL
    const webUrl = `https://github.com/${GITHUB_REPO}/blob/${GITHUB_BRANCH}/docs/docs/${mdxFile}`;

    try {
      const response = await fetch(rawUrl);
      if (!response.ok) throw new Error('File not found on GitHub');
      let markdown = await response.text();
      markdown += `\n\n---\n[View this file on GitHub](${webUrl})`;
      await navigator.clipboard.writeText(markdown);
      alert('Markdown copied to clipboard!');
    } catch (error) {
      console.error('Failed to copy markdown:', error);
      alert('Failed to copy markdown from GitHub.');
    }
  };

  return (
    <button 
      onClick={copyMarkdown}
      className="button button--secondary button--sm"
      style={{ marginLeft: '10px' }}
    >
      📋 Copy as Markdown
    </button>
  );
};

export default CopyMarkdownButton;