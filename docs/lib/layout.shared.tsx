import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import {
  BookOpen,
  Compass,
  GraduationCap,
  Blocks,
  History,
  Newspaper,
} from 'lucide-react';
import { appName, gitConfig } from './shared';

// Nav items rendered in the middle column of the top nav, between the
// logo and the search bar. Exported so our custom header slot
// (`src/components/NavHeader`) can consume it; deliberately NOT
// passed via Fumadocs' `links` option, because that flow places text
// items on the far right of the header — we want the classic "Logo |
// Nav — — Search | Icons" layout (Tailwind / Next.js docs style) with
// the items aligned under the main content column.
//
// Icons chosen for semantic clarity + visual distinction at 16px:
//   Docs         → BookOpen      (reading reference material)
//   Guides       → Compass       (directional walkthroughs)
//   Tutorials    → GraduationCap (learning path)
//   Integrations → Blocks        (modular pluggable pieces)
//   Changelog    → History       (time-ordered records)
//   Blog         → Newspaper     (articles / posts)
export const navLinks = [
  { text: 'Docs', url: '/docs/getting-started', activeBase: '/docs', icon: <BookOpen /> },
  { text: 'Guides', url: '/guides/guides-ai-agent-evaluation', activeBase: '/guides', icon: <Compass /> },
  { text: 'Tutorials', url: '/tutorials/tutorial-introduction', activeBase: '/tutorials', icon: <GraduationCap /> },
  { text: 'Integrations', url: '/integrations/models/openai', activeBase: '/integrations', icon: <Blocks /> },
  { text: 'Changelog', url: '/changelog', activeBase: '/changelog', icon: <History /> },
  { text: 'Blog', url: '/blog', activeBase: '/blog', icon: <Newspaper /> },
];

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span
          role="img"
          aria-label={appName}
          style={{
            display: 'block',
            height: '28px',
            width: '131px',
            backgroundColor: 'var(--color-fd-foreground)',
            WebkitMask: 'url("/icons/DeepEval.svg") no-repeat center / contain',
            mask: 'url("/icons/DeepEval.svg") no-repeat center / contain',
          }}
        />
      ),
      // NOTE: no `nav.children` here — the nav link strip is rendered
      // directly inside our custom header slot (`NavHeader`) so it
      // lands in the middle grid column, right under the main content.
      // Fumadocs would otherwise stash `children` next to `navTitle`
      // in the left cell, which is the wrong column.
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    // `links` intentionally omitted — text items live in `navLinks`
    // (rendered by `NavHeader`); only the GitHub icon flows through
    // Fumadocs' `navItems` via `githubUrl`, and our header picks it
    // up from `useNotebookLayout().navItems`.
  };
}
