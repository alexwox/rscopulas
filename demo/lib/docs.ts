import "server-only";

import fs from "node:fs/promises";
import path from "node:path";
import { cache } from "react";

import matter from "gray-matter";

type LogoConfig =
  | string
  | {
      light?: string;
      dark?: string;
      href?: string;
    };

type NavGroup = {
  group: string;
  pages: string[];
};

type NavTab = {
  tab: string;
  groups: NavGroup[];
};

type DocsConfig = {
  name: string;
  description?: string;
  logo?: LogoConfig;
  navigation: {
    tabs: NavTab[];
  };
};

export type DocNavEntry = {
  slug: string;
  segments: string[];
  tab: string;
  group: string;
  order: number;
};

export type DocPage = DocNavEntry & {
  title: string;
  sidebarTitle?: string;
  description?: string;
  body: string;
};

const getDocsRoot = cache(async () => {
  const candidates = [
    path.join(process.cwd(), "..", "docs", "mdx"),
    path.join(process.cwd(), "docs", "mdx"),
  ];

  for (const candidate of candidates) {
    try {
      await fs.access(path.join(candidate, "docs.json"));
      return candidate;
    } catch {
      // Try the next candidate.
    }
  }

  throw new Error("Could not locate docs/mdx with a docs.json file.");
});

export const getDocsConfig = cache(async (): Promise<DocsConfig> => {
  const docsRoot = await getDocsRoot();
  const source = await fs.readFile(path.join(docsRoot, "docs.json"), "utf8");
  return JSON.parse(source) as DocsConfig;
});

export const getDocNavEntries = cache(async (): Promise<DocNavEntry[]> => {
  const config = await getDocsConfig();
  const entries: DocNavEntry[] = [];
  let order = 0;

  for (const tab of config.navigation.tabs) {
    for (const group of tab.groups) {
      for (const slug of group.pages) {
        entries.push({
          slug,
          segments: slug.split("/"),
          tab: tab.tab,
          group: group.group,
          order,
        });
        order += 1;
      }
    }
  }

  return entries;
});

export const getHomeDoc = cache(async (): Promise<DocPage> => {
  const docsRoot = await getDocsRoot();
  const source = await fs.readFile(path.join(docsRoot, "index.mdx"), "utf8");
  const parsed = matter(source);
  const data = parsed.data as Record<string, unknown>;

  return {
    slug: "",
    segments: [],
    tab: "Home",
    group: "Home",
    order: -1,
    title: typeof data.title === "string" ? data.title : "Home",
    sidebarTitle:
      typeof data.sidebarTitle === "string" ? data.sidebarTitle : undefined,
    description:
      typeof data.description === "string" ? data.description : undefined,
    body: parsed.content,
  };
});

export const getDocPage = cache(async (slug: string): Promise<DocPage> => {
  const docsRoot = await getDocsRoot();
  const entry = (await getDocNavEntries()).find((item) => item.slug === slug);

  if (!entry) {
    throw new Error(`Unknown doc slug: ${slug}`);
  }

  const filePath = path.join(docsRoot, `${slug}.mdx`);
  const source = await fs.readFile(filePath, "utf8");
  const parsed = matter(source);
  const data = parsed.data as Record<string, unknown>;

  return {
    ...entry,
    title:
      typeof data.title === "string"
        ? data.title
        : entry.segments.at(-1)?.replace(/-/g, " ") ?? entry.slug,
    sidebarTitle:
      typeof data.sidebarTitle === "string" ? data.sidebarTitle : undefined,
    description:
      typeof data.description === "string" ? data.description : undefined,
    body: parsed.content,
  };
});

export async function getTabLinks() {
  const config = await getDocsConfig();

  return config.navigation.tabs
    .map((tab) => {
      const firstPage = tab.groups.flatMap((group) => group.pages)[0];
      if (!firstPage) {
        return null;
      }

      return {
        label: tab.tab,
        href: `/${firstPage}`,
      };
    })
    .filter((tab): tab is { label: string; href: string } => tab !== null);
}

export async function getPageNeighbors(slug: string) {
  const entries = await getDocNavEntries();
  const index = entries.findIndex((entry) => entry.slug === slug);

  return {
    previous: index > 0 ? entries[index - 1] : null,
    next: index >= 0 && index < entries.length - 1 ? entries[index + 1] : null,
  };
}
