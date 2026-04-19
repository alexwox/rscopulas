import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "rscopulas docs demo",
    template: "%s | rscopulas docs demo",
  },
  description: "A simple Next.js docs site powered by docs/mdx content.",
  icons: {
    icon: "/rscopulas-logo.png",
    shortcut: "/rscopulas-logo.png",
    apple: "/rscopulas-logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
