import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import 'katex/dist/katex.css';
import { Geist, Geist_Mono } from 'next/font/google';

const sans = Geist({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
});

const heading = Geist_Mono({
  subsets: ['latin'],
  variable: '--font-heading',
  display: 'swap',
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${heading.variable}`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen font-sans">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
