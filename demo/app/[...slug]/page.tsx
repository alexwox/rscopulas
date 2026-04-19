import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { DocsShell } from "@/components/docs-shell";
import { MdxContent } from "@/components/mdx-content";
import { getDocNavEntries, getDocPage } from "@/lib/docs";

type RouteParams = {
  slug: string[];
};

export const dynamicParams = false;

export async function generateStaticParams(): Promise<RouteParams[]> {
  const entries = await getDocNavEntries();
  return entries.map((entry) => ({
    slug: entry.segments,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<RouteParams>;
}): Promise<Metadata> {
  const { slug } = await params;
  const docSlug = slug.join("/");

  try {
    const page = await getDocPage(docSlug);
    return {
      title: page.title,
      description: page.description,
    };
  } catch {
    return {};
  }
}

export default async function DocPage({
  params,
}: {
  params: Promise<RouteParams>;
}) {
  const { slug } = await params;
  const docSlug = slug.join("/");

  try {
    const page = await getDocPage(docSlug);

    return (
      <DocsShell
        currentSlug={docSlug}
        title={page.title}
        description={page.description}
      >
        <MdxContent source={page.body} />
      </DocsShell>
    );
  } catch {
    notFound();
  }
}
