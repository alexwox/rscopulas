import Link from "next/link";

export default function NotFound() {
  return (
    <main className="not-found">
      <div>
        <p className="eyebrow">404</p>
        <h1>Page not found</h1>
        <p>This route does not exist in the `docs/mdx` navigation.</p>
        <Link href="/">Back to the docs home</Link>
      </div>
    </main>
  );
}
