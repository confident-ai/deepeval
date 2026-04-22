import { HomeLayout } from 'fumadocs-ui/layouts/home';
import { baseOptions } from '@/lib/layout.shared';
import Footer from '@/src/components/Footer';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <>
      <HomeLayout {...baseOptions()}>{children}</HomeLayout>
      <Footer />
    </>
  );
}
