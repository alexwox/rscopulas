import type { Metadata } from "next";

import { DocsShell } from "@/components/docs-shell";
import { MdxContent } from "@/components/mdx-content";
import { getHomeDoc } from "@/lib/docs";

export async function generateMetadata(): Promise<Metadata> {
  const home = await getHomeDoc();

  return {
    title: home.title,
    description: home.description,
  };
}

export default async function HomePage() {
  const home = await getHomeDoc();

  return (
    <DocsShell title={home.title} description={home.description}>
      <MdxContent source={home.body} />
    </DocsShell>
  );
}
