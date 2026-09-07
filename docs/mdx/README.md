# rscopulas user documentation

This directory is the canonical source for the user guides and API reference.
The Next.js site in `../../demo` renders these MDX files. `docs.json` describes
navigation and also supports Mintlify. Keep each page here once.

To preview from the repository root:

```sh
cd demo
npm ci
npm run dev
```

Run `npm run build` in `demo/` to compile every documentation page before
submitting changes. Headings receive stable fragment IDs through `rehype-slug`.
