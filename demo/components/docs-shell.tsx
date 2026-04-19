import type { ReactNode } from "react";

import Image from "next/image";
import Link from "next/link";

import { getDocsConfig, getDocNavEntries, getPageNeighbors, getTabLinks } from "@/lib/docs";

function getLogoSource(
  logo:
    | string
    | {
        light?: string;
        dark?: string;
        href?: string;
      }
    | undefined,
) {
  if (!logo) {
    return null;
  }

  if (typeof logo === "string") {
    return { src: logo, href: "/" };
  }

  return {
    src: logo.light ?? logo.dark ?? null,
    href: logo.href ?? "/",
  };
}

export async function DocsShell({
  currentSlug,
  title,
  description,
  children,
}: {
  currentSlug?: string;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  const [config, entries, tabLinks] = await Promise.all([
    getDocsConfig(),
    getDocNavEntries(),
    getTabLinks(),
  ]);

  const currentEntry = currentSlug
    ? entries.find((entry) => entry.slug === currentSlug) ?? null
    : null;
  const currentTab = currentEntry?.tab ?? config.navigation.tabs[0]?.tab;
  const visibleTab =
    config.navigation.tabs.find((tab) => tab.tab === currentTab) ??
    config.navigation.tabs[0];
  const neighbors = currentSlug ? await getPageNeighbors(currentSlug) : null;
  const logo = getLogoSource(config.logo);

  return (
    <div className="docs-shell">
      <header className="topbar">
        <div className="brand-row">
          <Link href={logo?.href ?? "/"} className="brand-link">
            {logo?.src ? (
              <Image
                src={logo.src}
                alt={config.name}
                width={36}
                height={36}
                className="brand-image"
                unoptimized
              />
            ) : null}
            <span>{config.name}</span>
          </Link>
          <nav className="tab-links" aria-label="Top navigation">
            <Link href="/" className={!currentSlug ? "tab-link active" : "tab-link"}>
              Home
            </Link>
            {tabLinks.map((tab) => (
              <Link
                key={tab.label}
                href={tab.href}
                className={currentTab === tab.label ? "tab-link active" : "tab-link"}
              >
                {tab.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <div className="docs-layout">
        <aside className="sidebar">
          {visibleTab?.groups.map((group) => (
            <section key={group.group} className="sidebar-group">
              <h2>{group.group}</h2>
              <ul>
                {group.pages.map((pageSlug) => {
                  const entry = entries.find((item) => item.slug === pageSlug);
                  const label = entry?.segments.at(-1)?.replace(/-/g, " ") ?? pageSlug;

                  return (
                    <li key={pageSlug}>
                      <Link
                        href={`/${pageSlug}`}
                        className={currentSlug === pageSlug ? "sidebar-link active" : "sidebar-link"}
                      >
                        {label}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </aside>

        <main className="content">
          <article className="prose">
            <header className="page-header">
              <h1>{title}</h1>
              {description ? <p>{description}</p> : null}
            </header>
            {children}
          </article>

          {neighbors ? (
            <nav className="pager" aria-label="Page navigation">
              {neighbors.previous ? (
                <Link href={`/${neighbors.previous.slug}`} className="pager-link">
                  <span>Previous</span>
                  <strong>{neighbors.previous.slug}</strong>
                </Link>
              ) : (
                <div />
              )}
              {neighbors.next ? (
                <Link href={`/${neighbors.next.slug}`} className="pager-link align-right">
                  <span>Next</span>
                  <strong>{neighbors.next.slug}</strong>
                </Link>
              ) : null}
            </nav>
          ) : null}
        </main>
      </div>
    </div>
  );
}
