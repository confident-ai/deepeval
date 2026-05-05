import HomePageShell from "@/src/layouts/HomePageShell";

export default function Layout({ children }: LayoutProps<"/">) {
  return <HomePageShell>{children}</HomePageShell>;
}
