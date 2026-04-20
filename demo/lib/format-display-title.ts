function capitalizeWord(word: string): string {
  if (!word) {
    return word;
  }

  return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

/** Title-case labels derived from slugs (e.g. "vine-copula" → "Vine Copula"). */
export function formatDisplayTitle(raw: string): string {
  const normalized = raw.replace(/[-_]+/g, " ").trim();
  if (!normalized) {
    return raw;
  }

  return normalized
    .split(/\s+/)
    .filter(Boolean)
    .map(capitalizeWord)
    .join(" ");
}
