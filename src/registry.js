const MANIFEST_URL = `${import.meta.env.BASE_URL}artifacts-manifest.json`;

let manifestPromise = null;

export function getManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(MANIFEST_URL).then((res) => {
      if (!res.ok) throw new Error(`Failed to load artifact manifest (${res.status})`);
      return res.json();
    });
  }
  return manifestPromise;
}

export function categoriesOf(manifest) {
  const counts = new Map();
  for (const a of manifest) counts.set(a.category, (counts.get(a.category) || 0) + 1);
  return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}
