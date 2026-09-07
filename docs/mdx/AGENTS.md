# Documentation instructions

- `docs/mdx` is the canonical source for rscopulas user guides and API reference.
- The Next.js renderer lives in `demo/`; `docs.json` defines navigation for Mintlify compatibility.
- Keep MDX frontmatter and existing component conventions.
- Use concise sentences, active voice, and sentence case headings.
- Verify behavior against the Rust core and Python wrappers. Distinguish joint likelihood from composite scores and measured checks from unsupported capabilities.
- Update API examples when signatures or semantics change; do not create duplicate documentation trees.
- Validate all pages with `npm run build` from `demo/`.
