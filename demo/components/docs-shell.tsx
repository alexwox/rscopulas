import type { ReactNode } from "react";

import Image from "next/image";
import Link from "next/link";

import { getDocsConfig, getDocNavEntries, getPageNeighbors, getTabLinks } from "@/lib/docs";
import { formatDisplayTitle } from "@/lib/format-display-title";
import { ThemeToggle } from "@/components/theme-toggle";
import { MobileNav } from "@/components/mobile-nav";

const DEMO_LOGO_SRC = "/rscopulas-logo.png";

const EXTERNAL_LINKS = [
  {
    href: "https://github.com/alexwox/rscopulas",
    label: "GitHub",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.57.1.78-.25.78-.55v-2c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.18 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.77.12 3.06.74.81 1.19 1.84 1.19 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.2.66.79.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z" />
      </svg>
    ),
  },
  {
    href: "https://pypi.org/project/rscopulas/",
    label: "PyPI",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M11.9 0a8.7 8.7 0 0 0-2.2.3C7.6.8 7.2 2 7.2 4v2.4h4.8v.6H4.7c-2 0-3.7 1.2-4.2 3.4a13 13 0 0 0 0 6c.5 2.1 1.8 3.4 3.8 3.4H6v-2.8c0-2.2 1.9-4.2 4.2-4.2h4.8c1.9 0 3.4-1.6 3.4-3.5V4c0-1.9-1.6-3.3-3.4-3.6A22 22 0 0 0 11.9 0zM9.3 1.5a1.1 1.1 0 1 1 0 2.2 1.1 1.1 0 0 1 0-2.2zm8.6 5.5v2.7c0 2.3-2 4.3-4.2 4.3H8.9c-1.9 0-3.4 1.6-3.4 3.5V23c0 1.9 1.6 3 3.4 3.5 2.1.6 4.2.7 6.8 0 1.7-.5 3.4-1.5 3.4-3.5v-2.4h-4.8V20h7.2c2 0 2.8-1.5 3.5-3.6a13 13 0 0 0 0-6c-.5-2.1-1.4-3.4-3.5-3.4h-3.6zm-2.7 11.6a1.1 1.1 0 1 1 0 2.2 1.1 1.1 0 0 1 0-2.2z" />
      </svg>
    ),
  },
  {
    href: "https://crates.io/crates/rscopulas",
    label: "crates.io",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
    ),
  },
];

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
  const logoHref =
    typeof config.logo === "object" && config.logo?.href ? config.logo.href : "/";

  const tabNav = (
    <nav className="tab-links" aria-label="Top navigation">
      <Link href="/" className={!currentSlug ? "tab-link active" : "tab-link"}>
        Home
      </Link>
      {tabLinks.map((tab) => (
        <Link
          key={tab.label}
          href={tab.href}
          className={currentSlug && currentTab === tab.label ? "tab-link active" : "tab-link"}
        >
          {tab.label}
        </Link>
      ))}
    </nav>
  );

  const sidebarNav = (
    <>
      {visibleTab?.groups.map((group) => (
        <section key={group.group} className="sidebar-group">
          <h2>{group.group}</h2>
          <ul>
            {group.pages.map((pageSlug) => {
              const entry = entries.find((item) => item.slug === pageSlug);
              const label = formatDisplayTitle(
                entry?.segments.at(-1)?.replace(/-/g, " ") ?? pageSlug,
              );

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
    </>
  );

  const externalIconLinks = EXTERNAL_LINKS.map((link) => (
    <a
      key={link.href}
      href={link.href}
      target="_blank"
      rel="noreferrer"
      className="icon-link"
      aria-label={link.label}
      title={link.label}
    >
      {link.icon}
    </a>
  ));

  return (
    <div className="docs-shell">
      <header className="topbar">
        <div className="brand-row">
          <Link href={logoHref} className="brand-link">
            <Image
              src={DEMO_LOGO_SRC}
              alt={config.name}
              width={40}
              height={40}
              className="brand-image"
            />
            <span>{config.name}</span>
          </Link>
          <div className="topbar-actions">
            <div className="topbar-tabs">{tabNav}</div>
            <div className="topbar-icons">
              {externalIconLinks}
              <ThemeToggle />
            </div>
            <MobileNav>
              <div className="mobile-nav-section">
                <p className="mobile-nav-section-label">Navigate</p>
                {tabNav}
              </div>
              <div className="mobile-nav-section mobile-nav-sidebar">
                <p className="mobile-nav-section-label">
                  {visibleTab?.tab ?? "Documentation"}
                </p>
                {sidebarNav}
              </div>
              <div className="mobile-nav-section mobile-nav-icons">
                {externalIconLinks}
              </div>
            </MobileNav>
          </div>
        </div>
      </header>

      <div className="docs-layout">
        <aside className="sidebar">{sidebarNav}</aside>

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
                  <strong>
                    {formatDisplayTitle(
                      neighbors.previous.segments.at(-1)?.replace(/-/g, " ") ??
                        neighbors.previous.slug,
                    )}
                  </strong>
                </Link>
              ) : (
                <div />
              )}
              {neighbors.next ? (
                <Link href={`/${neighbors.next.slug}`} className="pager-link align-right">
                  <span>Next</span>
                  <strong>
                    {formatDisplayTitle(
                      neighbors.next.segments.at(-1)?.replace(/-/g, " ") ??
                        neighbors.next.slug,
                    )}
                  </strong>
                </Link>
              ) : null}
            </nav>
          ) : null}
        </main>
      </div>
    </div>
  );
}
